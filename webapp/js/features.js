/**
 * Turning a user ID into the nine features the model needs.
 *
 * The honest constraint: a static page cannot fetch a real Instagram profile.
 * The Graph API only exposes accounts that authorised your app, scraping is
 * blocked by CORS and violates Instagram's terms, and there is no server here to
 * proxy either. So a bare user ID cannot produce real statistics.
 *
 * Rather than hide that, this module generates a *deterministic demo profile*
 * from the username and flags it. `Analyze` still works from an ID alone — the
 * flow the app is built around — but the result is stamped as demo data and the
 * generated numbers are shown in full, so nobody mistakes it for a real check.
 * Fill the profile fields in by hand and the flag disappears.
 */

const Features = (() => {
  'use strict';

  // Column names exactly as the model was fitted on.
  const COLUMNS = {
    followers: 'Followers',
    following: 'Following',
    posts: 'Posts',
    engagement: 'Engagement Rate (%)',
    likes: 'Avg Likes per Post',
    comments: 'Avg Comments per Post',
    verified: 'Verified',
    age: 'Account Age (Years)',
    bio: 'Bio Text',
  };

  const LABELS = {
    'Followers': 'Followers',
    'Following': 'Following',
    'Posts': 'Posts',
    'Engagement Rate (%)': 'Engagement rate',
    'Avg Likes per Post': 'Avg likes / post',
    'Avg Comments per Post': 'Avg comments / post',
    'Verified': 'Verified badge',
    'Account Age (Years)': 'Account age',
    'Bio Text': 'Bio text',
  };

  const DEMO_BIOS = [
    'Foodie | Reviews and recipes',
    'Photographer and content creator',
    'Digital nomad living my best life',
    'DM for promo!! Click link in bio',
    'FREE FOLLOWERS >>> link below',
    'Fitness enthusiast | Health advocate',
    'Follow back 100% guaranteed',
    'Just here for the dog photos',
  ];

  /**
   * FNV-1a, 32-bit. Deterministic and dependency-free: the same username always
   * produces the same demo profile, so demos and bug reports are reproducible.
   */
  function hash(text, seed) {
    let h = (2166136261 ^ seed) >>> 0;
    for (let i = 0; i < text.length; i++) {
      h ^= text.charCodeAt(i);
      h = Math.imul(h, 16777619) >>> 0;
    }
    return h;
  }

  const pick = (name, seed, lo, hi) => lo + (hash(name, seed) % Math.max(1, hi - lo));

  return {
    COLUMNS,
    LABELS,

    /** Human label for a model column. */
    label(column) { return LABELS[column] || column; },

    /** Instagram handle rules: letters, digits, dots, underscores, 1-30 chars. */
    validateUserId(value) {
      const id = String(value || '').trim().replace(/^@/, '');
      if (!id) return { ok: false, error: 'Enter a user ID.' };
      if (id.length > 30) return { ok: false, error: 'User IDs are at most 30 characters.' };
      if (!/^[A-Za-z0-9._]+$/.test(id)) {
        return { ok: false, error: 'Only letters, numbers, dots and underscores are allowed.' };
      }
      return { ok: true, value: id };
    },

    /**
     * Deterministic demo profile for a username.
     * The numbers are internally consistent (likes track followers and the
     * engagement rate) so the model sees a plausible row rather than noise.
     */
    demoProfile(userId) {
      const name = userId.toLowerCase();
      const followers = pick(name, 1, 80, 60000);
      const engagement = (pick(name, 2, 20, 1200) / 100);
      const likes = Math.round(followers * engagement / 100);
      const comments = Math.round(likes * (pick(name, 3, 1, 60) / 1000));

      return {
        [COLUMNS.followers]: followers,
        [COLUMNS.following]: pick(name, 4, 20, 6000),
        [COLUMNS.posts]: pick(name, 5, 0, 800),
        [COLUMNS.engagement]: Number(engagement.toFixed(2)),
        [COLUMNS.likes]: likes,
        [COLUMNS.comments]: comments,
        [COLUMNS.verified]: hash(name, 6) % 100 < 8 ? 1 : 0,
        [COLUMNS.age]: pick(name, 7, 0, 13),
        [COLUMNS.bio]: DEMO_BIOS[hash(name, 8) % DEMO_BIOS.length],
      };
    },

    /**
     * Read and validate the manual profile form.
     * @returns {{ok:boolean, profile?:Object, errors?:Object<string,string>}}
     */
    readForm(form) {
      const errors = {};
      const profile = {};

      const numeric = [
        ['followers', COLUMNS.followers, 0, 1e10, true],
        ['following', COLUMNS.following, 0, 1e10, true],
        ['posts', COLUMNS.posts, 0, 1e7, true],
        ['engagement', COLUMNS.engagement, 0, 100, false],
        ['likes', COLUMNS.likes, 0, 1e9, true],
        ['comments', COLUMNS.comments, 0, 1e9, true],
        ['age', COLUMNS.age, 0, 100, false],
      ];

      for (const [field, column, min, max, isInt] of numeric) {
        const raw = String(form[field] ?? '').trim();
        if (raw === '') { errors[field] = 'Required.'; continue; }
        const value = Number(raw);
        if (!Number.isFinite(value)) { errors[field] = 'Must be a number.'; continue; }
        if (value < min || value > max) {
          errors[field] = `Must be between ${min} and ${max.toLocaleString()}.`;
          continue;
        }
        profile[column] = isInt ? Math.round(value) : value;
      }

      profile[COLUMNS.verified] = form.verified ? 1 : 0;

      const bio = String(form.bio ?? '').trim();
      if (!bio) errors.bio = 'Required. Use "(no bio)" if the profile has none.';
      else if (bio.length > 2000) errors.bio = 'At most 2000 characters.';
      else profile[COLUMNS.bio] = bio;

      return Object.keys(errors).length ? { ok: false, errors } : { ok: true, profile };
    },

    /** Format a feature value for display. */
    format(column, value) {
      if (column === COLUMNS.bio) return String(value || '—');
      if (column === COLUMNS.verified) return Number(value) >= 0.5 ? 'Yes' : 'No';
      if (column === COLUMNS.engagement) return `${Number(value).toFixed(2)}%`;
      if (column === COLUMNS.age) {
        const years = Number(value);
        return years === 1 ? '1 year' : `${years.toFixed(1).replace(/\.0$/, '')} years`;
      }
      return Math.round(Number(value)).toLocaleString();
    },
  };
})();
