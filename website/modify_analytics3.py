import re

def main():
    template_path = 'templates/analytics.html'
    with open(template_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 1. Insert variable declarations after the opening <script> tag.
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        new_lines.append(line)
        # Check if this line contains the opening <script> tag (not the src one)
        if '<script>' in line and not ('src=' in line):
            # Determine indentation of the next line (the line after this script tag)
            # We'll use the same indentation as the next non-empty line that starts with whitespace.
            j = i + 1
            while j < len(lines) and lines[j].strip() == '':
                j += 1
            indent = ''
            if j < len(lines):
                # capture leading spaces/tabs
                match = re.match(r'(\s*)', lines[j])
                if match:
                    indent = match.group(1)
            # Insert variable declaration lines with same indentation
            vars_block = [
                '// Chart data from Flask',
                'const feature_names = {{ feature_names_json|safe }};',
                'const feature_importances = {{ feature_importances|safe }};',
                'const fake_count = {{ fake_count|safe }};',
                'const real_count = {{ real_count|safe }};',
                'const prediction_percentages = {{ prediction_percentages|safe }};',
                'const engagement_bins = {{ engagement_bins|safe }};',
                'const engagement_counts = {{ engagement_counts|safe }};',
                ''
            ]
            for vb in vars_block:
                if vb == '':
                    # add empty line to keep spacing
                    new_lines.append('\n')
                else:
                    new_lines.append(indent + vb + '\n')
            # Continue loop; the next iteration will add the original line (which we already added? Actually we already added the line when we appended line.)
            # We have already added the line containing <script>. We want to keep the following lines as they are.
            # So we do not skip the next line; we just continue.
        i += 1

    # Join back to string for easier replacement of specific patterns
    content = ''.join(new_lines)

    # 2. Replace labels and data in feature chart
    # We'll replace the line that contains "labels: ['Followers', ...]" with "labels: feature_names,"
    # We'll do line-by-line again to preserve indentation.
    lines = content.splitlines(keepends=True)
    out_lines = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('labels: [') and 'Followers' in stripped:
            indent = len(line) - len(linediff := line.lstrip())
            # Actually compute indent:
            indent = len(line) - len(linediff)
            out_lines.append(' ' * indent + 'labels: feature_names,')
        elif stripped.startswith('data: [') and '0.18' in stripped:
            indent = len(line) - len(line.lstrip())
            out_lines.append(' ' * indent + 'data: feature_importances,')
        elif stripped.startswith('backgroundColor: ['):
            indent = len(line) - len(line.lstrip())
            out_lines.append(' ' * indent + "backgroundColor: 'rgba(110, 142, 251, 0.8)',")
        elif stripped.startswith('borderColor: ['):
            indent = len(line) - len(line.lstrip())
            out_lines.append(' ' * indent + "borderColor: 'rgba(110, 142, 251, 1)',")
        else:
            out_lines.append(line)
    content = ''.join(out_lines)

    # 3. Replace pie chart data
    lines = content.splitlines(keepends=True)
    out_lines = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('data: [') and '342' in stripped and '898' in stripped:
            indent = len(line) - len(linediff := line.lstrip())
            # Actually compute:
            indent = len(line) - len(linediff)
            out_lines.append(' ' * indent + 'data: [fake_count, real_count],')
        else:
            out_lines.append(line)
    content = ''.join(out_lines)

    # Write back
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('Updated analytics.html with dynamic chart data.')

if __name__ == '__main__':
    main()