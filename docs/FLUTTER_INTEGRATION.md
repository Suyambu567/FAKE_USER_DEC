# Flutter Integration Guide

How a Flutter app (Android / iOS / Web) consumes the Fake Profile Detection API.

---

## 1. Base URL

| Environment | Base URL | Note |
|---|---|---|
| Android emulator → host machine | `http://10.0.2.2:8000` | `localhost` inside the emulator is the emulator itself |
| iOS simulator → host machine | `http://127.0.0.1:8000` | shares the host network |
| Physical device on LAN | `http://<your-lan-ip>:8000` | needs a cleartext exception, see §7 |
| Production | `https://api.example.com` | TLS terminated at the reverse proxy |

All paths below are relative to `<base>/api/v1`.

```dart
class ApiConfig {
  static const String baseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: 'http://10.0.2.2:8000',
  );
  static const String apiKey = String.fromEnvironment('API_KEY', defaultValue: '');
  static const Duration timeout = Duration(seconds: 20);
}
```

Build with `flutter run --dart-define=API_BASE_URL=https://api.example.com --dart-define=API_KEY=...`.
Never hardcode a production URL in source.

---

## 2. The response envelope

Every endpoint returns the same shape, on success and on failure:

```json
{
  "success": true,
  "message": "Prediction complete.",
  "data": { },
  "error": null,
  "request_id": "2e794745cc9c49ec"
}
```

```dart
class ApiResponse<T> {
  final bool success;
  final String message;
  final T? data;
  final String? errorCode;
  final List<FieldError> fieldErrors;
  final String? requestId;

  const ApiResponse({
    required this.success,
    required this.message,
    this.data,
    this.errorCode,
    this.fieldErrors = const [],
    this.requestId,
  });

  factory ApiResponse.fromJson(
    Map<String, dynamic> json,
    T Function(Map<String, dynamic>)? parse,
  ) {
    final error = json['error'] as Map<String, dynamic>?;
    final details = error?['details'];

    return ApiResponse(
      success: json['success'] as bool? ?? false,
      message: json['message'] as String? ?? '',
      data: (json['data'] != null && parse != null)
          ? parse(json['data'] as Map<String, dynamic>)
          : null,
      errorCode: error?['code'] as String?,
      fieldErrors: details is List
          ? details
              .whereType<Map<String, dynamic>>()
              .map(FieldError.fromJson)
              .toList()
          : const [],
      requestId: json['request_id'] as String?,
    );
  }
}

class FieldError {
  final String field;
  final String message;
  const FieldError(this.field, this.message);

  factory FieldError.fromJson(Map<String, dynamic> j) =>
      FieldError(j['field'] as String? ?? '', j['message'] as String? ?? '');
}
```

**Branch on `errorCode`, never on `message`.** Codes are contract; messages are copy and will change.

| `error.code` | HTTP | What the app should do |
|---|---|---|
| `validation_error` | 422 | Show `fieldErrors` inline on the form |
| `unauthorized` | 401 | Key is wrong/missing — do not retry |
| `rate_limited` | 429 | Back off for `Retry-After` seconds, then retry |
| `model_not_ready` | 503 | Retry with backoff; another instance may serve |
| `inference_timeout` | 504 | Retry once, then surface a failure |
| `internal_error` | 500 | Show `request_id` in the error UI for support |

---

## 3. Models

```dart
class ProfileFeatures {
  final int followers, following, posts, avgLikesPerPost, avgCommentsPerPost;
  final double engagementRate, accountAgeYears;
  final bool verified;
  final String bioText;

  const ProfileFeatures({
    required this.followers,
    required this.following,
    required this.posts,
    required this.engagementRate,
    required this.avgLikesPerPost,
    required this.avgCommentsPerPost,
    required this.verified,
    required this.accountAgeYears,
    required this.bioText,
  });

  Map<String, dynamic> toJson() => {
        'followers': followers,
        'following': following,
        'posts': posts,
        'engagement_rate': engagementRate,
        'avg_likes_per_post': avgLikesPerPost,
        'avg_comments_per_post': avgCommentsPerPost,
        'verified': verified,
        'account_age_years': accountAgeYears,
        'bio_text': bioText,
      };
}

class Prediction {
  final String label;                      // "Fake" | "Real"
  final double confidence;                 // 0..1
  final Map<String, double> probabilities;
  final String modelVersion;
  final double latencyMs;

  const Prediction({
    required this.label,
    required this.confidence,
    required this.probabilities,
    required this.modelVersion,
    required this.latencyMs,
  });

  factory Prediction.fromJson(Map<String, dynamic> j) => Prediction(
        label: j['label'] as String,
        confidence: (j['confidence'] as num).toDouble(),
        probabilities: (j['probabilities'] as Map<String, dynamic>)
            .map((k, v) => MapEntry(k, (v as num).toDouble())),
        modelVersion: j['model_version'] as String,
        latencyMs: (j['latency_ms'] as num).toDouble(),
      );

  bool get isFake => label == 'Fake';
}
```

---

## 4. The client

Uses `package:http`. Swap in `dio` if you want interceptors — the contract is identical.

```yaml
# pubspec.yaml
dependencies:
  http: ^1.2.2
```

```dart
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ApiException implements Exception {
  final String code;
  final String message;
  final String? requestId;
  final List<FieldError> fieldErrors;
  final int? statusCode;
  ApiException(this.code, this.message,
      {this.fieldErrors = const [], this.requestId, this.statusCode});
  @override
  String toString() => 'ApiException($code): $message';
}

class FakeProfileApi {
  FakeProfileApi({http.Client? client}) : _client = client ?? http.Client();
  final http.Client _client;

  Map<String, String> get _headers => {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        if (ApiConfig.apiKey.isNotEmpty) 'X-API-Key': ApiConfig.apiKey,
      };

  Uri _uri(String path) => Uri.parse('${ApiConfig.baseUrl}/api/v1$path');

  Future<T> _send<T>(
    Future<http.Response> Function() request,
    T Function(Map<String, dynamic>) parse,
  ) async {
    late http.Response res;
    try {
      res = await request().timeout(ApiConfig.timeout);
    } on SocketException {
      throw ApiException('network_error', 'No connection to the server.');
    } catch (_) {
      throw ApiException('timeout', 'The request timed out.');
    }

    Map<String, dynamic> body;
    try {
      body = jsonDecode(utf8.decode(res.bodyBytes)) as Map<String, dynamic>;
    } catch (_) {
      throw ApiException('bad_response', 'Malformed response from server.',
          statusCode: res.statusCode);
    }

    final envelope = ApiResponse<T>.fromJson(body, parse);
    if (!envelope.success || envelope.data == null) {
      throw ApiException(
        envelope.errorCode ?? 'unknown_error',
        envelope.message,
        fieldErrors: envelope.fieldErrors,
        requestId: envelope.requestId,
        statusCode: res.statusCode,
      );
    }
    return envelope.data as T;
  }

  Future<Prediction> predict(ProfileFeatures features) => _send(
        () => _client.post(_uri('/predict'),
            headers: _headers, body: jsonEncode(features.toJson())),
        Prediction.fromJson,
      );

  /// Prefer this whenever you have more than one profile: ~74x more efficient
  /// per profile than looping over `predict`.
  Future<List<Prediction>> predictBatch(List<ProfileFeatures> items) async {
    assert(items.isNotEmpty && items.length <= 100, 'batch must be 1..100');
    final data = await _send(
      () => _client.post(_uri('/predict/batch'),
          headers: _headers,
          body: jsonEncode({'items': items.map((e) => e.toJson()).toList()})),
      (j) => j,
    );
    return (data['results'] as List)
        .cast<Map<String, dynamic>>()
        .map((r) => Prediction(
              label: r['label'] as String,
              confidence: (r['confidence'] as num).toDouble(),
              probabilities: (r['probabilities'] as Map<String, dynamic>)
                  .map((k, v) => MapEntry(k, (v as num).toDouble())),
              modelVersion: data['model_version'] as String,
              latencyMs: (data['latency_ms'] as num).toDouble(),
            ))
        .toList();
  }

  Future<Map<String, dynamic>> modelInfo() =>
      _send(() => _client.get(_uri('/model/info'), headers: _headers), (j) => j);

  Future<Map<String, dynamic>> analytics() =>
      _send(() => _client.get(_uri('/analytics'), headers: _headers), (j) => j);

  Future<bool> isHealthy() async {
    try {
      final r = await _client
          .get(Uri.parse('${ApiConfig.baseUrl}/health/ready'))
          .timeout(const Duration(seconds: 5));
      return r.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  void dispose() => _client.close();
}
```

---

## 5. Handling errors in the UI

```dart
Future<void> _onAnalyse() async {
  setState(() { _loading = true; _fieldErrors = {}; });
  try {
    final result = await api.predict(_buildFeatures());
    if (!mounted) return;
    Navigator.push(context, MaterialPageRoute(
      builder: (_) => ResultScreen(prediction: result),
    ));
  } on ApiException catch (e) {
    if (!mounted) return;
    switch (e.code) {
      case 'validation_error':
        setState(() {
          _fieldErrors = { for (final f in e.fieldErrors) f.field: f.message };
        });
      case 'rate_limited':
        _snack('Too many requests. Try again in a moment.');
      case 'model_not_ready':
        _snack('The service is starting up. Please retry shortly.');
      case 'unauthorized':
        _snack('App is not authorised. Please update the app.');
      default:
        _snack('Something went wrong. Reference: ${e.requestId ?? "n/a"}');
    }
  } finally {
    if (mounted) setState(() => _loading = false);
  }
}
```

Wire `_fieldErrors[fieldName]` into each `TextFormField`'s `errorText` — the server field names
match the JSON keys exactly (`followers`, `bio_text`, …).

### Retry with backoff

Only `rate_limited`, `model_not_ready`, `timeout` and `network_error` are worth retrying.
`validation_error` and `unauthorized` never are.

```dart
Future<T> withRetry<T>(Future<T> Function() op, {int attempts = 3}) async {
  const retryable = {'rate_limited', 'model_not_ready', 'timeout', 'network_error'};
  for (var i = 0; ; i++) {
    try {
      return await op();
    } on ApiException catch (e) {
      if (i >= attempts - 1 || !retryable.contains(e.code)) rethrow;
      await Future.delayed(Duration(milliseconds: 400 * (1 << i)));  // 400/800/1600ms
    }
  }
}
```

---

## 6. Showing the verdict honestly

`GET /api/v1/model/info` returns `metrics.accuracy`, `metrics.baseline_accuracy`,
`metrics.lift_over_baseline` and a `warnings` array. **Gate your result UI on it.**

With the dataset currently shipped, `lift_over_baseline` is ~0.015 — statistically
indistinguishable from a coin flip. Presenting "Fake — 50.4% confidence" as a verdict is
misleading. Fetch the metadata once at app start and cache it:

```dart
final info = await api.modelInfo();
final metrics = info['metrics'] as Map<String, dynamic>;
final lift = (metrics['lift_over_baseline'] as num).toDouble();
final warnings = (info['warnings'] as List).cast<String>();

// Below ~5 points of lift the output is not decision-grade.
final modelIsUsable = lift > 0.05;
```

```dart
if (!modelIsUsable) {
  return const Banner(
    message: 'PREVIEW',
    location: BannerLocation.topEnd,
    child: Card(
      color: Colors.amber,
      child: ListTile(
        leading: Icon(Icons.warning_amber),
        title: Text('Preview only'),
        subtitle: Text(
          'This model is not accurate enough to judge real accounts. '
          'Results are for demonstration.',
        ),
      ),
    ),
  );
}
```

---

## 7. Platform setup

### Android — cleartext HTTP in development only

`android/app/src/main/AndroidManifest.xml`:

```xml
<application
    android:networkSecurityConfig="@xml/network_security_config"
    ...>
```

`android/app/src/main/res/xml/network_security_config.xml`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<network-security-config>
    <!-- Production traffic must be HTTPS. -->
    <base-config cleartextTrafficPermitted="false" />
    <domain-config cleartextTrafficPermitted="true">
        <domain includeSubdomains="true">10.0.2.2</domain>
        <domain includeSubdomains="true">localhost</domain>
    </domain-config>
</network-security-config>
```

Also add the internet permission:

```xml
<uses-permission android:name="android.permission.INTERNET" />
```

### iOS — ATS exception for local development

`ios/Runner/Info.plist`:

```xml
<key>NSAppTransportSecurity</key>
<dict>
    <key>NSAllowsLocalNetworking</key>
    <true/>
</dict>
```

Do **not** add `NSAllowsArbitraryLoads` — App Review rejects it without justification.

### Web (future ready)

Android and iOS ignore CORS; Flutter Web does not. Set `CORS_ORIGINS` on the server to your
exact web origin (e.g. `https://app.example.com`), not `*`, once you ship a web build.

---

## 8. Authentication

The API uses a shared key in the `X-API-Key` header. It is *optional* — leave `API_KEY` unset
on the server and no header is needed.

```
X-API-Key: <key>
```

**A key compiled into a Flutter binary is extractable.** Treat it as a throttling and
attribution control, not a secret. If you need real per-user identity — saved scans, quotas,
billing — replace `require_api_key` with OAuth2/JWT server-side; every protected route already
depends on that one function, so it is a single-file change. The Flutter side then becomes:

```dart
// after login
'Authorization': 'Bearer $accessToken',
```

with refresh-on-401 in an interceptor.

---

## 9. Not applicable

* **File upload** — the API takes JSON only; there are no upload endpoints.
* **WebSocket / streaming** — inference is a sub-second unary call; streaming would add
  complexity for no gain. If bulk scanning is ever added, prefer a job-submit + poll pattern
  over a socket.

---

## 10. Endpoint reference

### `POST /api/v1/predict`

Request:
```json
{
  "followers": 5000, "following": 300, "posts": 150,
  "engagement_rate": 4.5, "avg_likes_per_post": 400,
  "avg_comments_per_post": 20, "verified": false,
  "account_age_years": 5, "bio_text": "Foodie | Reviews and recipes"
}
```

Response `200`:
```json
{
  "success": true,
  "message": "Prediction complete.",
  "data": {
    "label": "Fake",
    "confidence": 0.50382,
    "probabilities": { "Fake": 0.50382, "Real": 0.49618 },
    "model_version": "20260807043949",
    "latency_ms": 86.03
  },
  "error": null,
  "request_id": "2e794745cc9c49ec"
}
```

Response `422`:
```json
{
  "success": false,
  "message": "The request payload is invalid.",
  "data": null,
  "error": {
    "code": "validation_error",
    "details": [{ "field": "followers", "message": "Input should be greater than or equal to 0" }]
  },
  "request_id": "..."
}
```

### `POST /api/v1/predict/batch`

Request: `{ "items": [ <ProfileFeatures>, … ] }` — 1 to 100 items.

Response `data`: `{ "results": [{ "index": 0, "label": "...", "confidence": 0.5, "probabilities": {…} }], "count": 1, "model_version": "...", "latency_ms": 65.66 }`

`results[i].index` maps back to `items[i]`; order is preserved.

### `GET /api/v1/features`

Returns the field spec (name, type, required, min, max, description) so the form can be built
from the server contract rather than hardcoded.

### `GET /api/v1/model/info`

Returns `model_version`, `algorithm`, `trained_at`, `sklearn_version`, `feature_count`,
`classes`, `metrics` and `warnings`.

### `GET /api/v1/analytics`

Returns `training_samples`, `feature_count`, `class_distribution[]`, `feature_importances[]`,
`engagement_histogram[]`, `metrics`. Precomputed at startup — cheap to poll.

---

## 11. Checklist before release

- [ ] `API_BASE_URL` supplied via `--dart-define`, HTTPS in production
- [ ] `cleartextTrafficPermitted` restricted to dev hosts only
- [ ] Every `ApiException` path has a user-visible message
- [ ] `request_id` shown in the generic error UI
- [ ] Batch endpoint used wherever more than one profile is scored
- [ ] `model_info.warnings` surfaced — no bare verdict from a low-lift model
- [ ] Retry/backoff wired for `rate_limited` and `model_not_ready`
- [ ] `CORS_ORIGINS` narrowed server-side if a web build ships
