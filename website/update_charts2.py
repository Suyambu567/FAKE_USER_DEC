import re
import sys

def main():
    template_path = 'templates/analytics.html'
    with open(template_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Insert variable declarations right after the opening <script> tag that follows the Chart.js script.
    # Pattern: '<script>' followed by newline maybe.
    # We'll find the first '<script>' after the Chart.js src line.
    # Simpler: replace '<script>\n        // Feature Importance Bar Chart' with '<script>\n        // Chart data from Flask\n        ...\n        // Feature Importance Bar Chart'
    # But we need to keep the indentation.

    # Let's split lines.
    lines = content.splitlines(keepends=True)
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        new_lines.append(line)
        # Look for the line that exactly contains '<script>' (with possible leading spaces)
        if line.rstrip() == '<script>':
            # Insert after this line
            # Add variable declarations with same indentation (8 spaces? Let's see)
            # The next line starts with 8 spaces then '// Feature Importance Bar Chart'
            # So we will insert 8 spaces plus our lines.
            indent = ' ' * 8  # assume 8 spaces
            var_lines = [
                f'{indent}// Chart data from Flask',
                f'{indent}const feature_names = {feature_json}|safe;',
                f'{indent}const feature_importances = {importance_json}|safe;',
                f'{indent}const fake_count = {fake_count}|safe;',
                f'{indent}const real_count = {real_count}|safe;',
                f'{indent}const prediction_percentages = {prediction_percentages}|safe;',
                f'{indent}const engagement_bins = {engagement_bins}|safe;',
                f'{indent}const engagement_counts = {engagement_counts}|safe;',
                f'{indent}// Feature Importance Bar Chart'
            ]
            # But we need to actually embed the Jinja syntax, not replace placeholders.
            # We'll write the actual Jinja expressions.
            # We'll define them later; for now just placeholders.
            # Actually we need to write the real template code.
            # Let's just write the lines with the proper Jinja syntax.
            # We'll do that after we have the variables defined as strings.
            # We'll construct the lines with the proper Jinja syntax.
            # We'll need to escape curly braces? No, we are writing to a .html file that will be processed by Flask later.
            # So we can write directly: {{ feature_names_json|safe }}
            # So we will write those.
            # Let's construct:
            #   const feature_names = {{ feature_names_json|safe }};
            # etc.
            # We'll add them after the script opening line.
            # But note we already added the line we are looking at; we need to insert before the existing comment line.
            # Since we have already appended the '<script>' line, we will now add our variable lines, then continue (the loop will add the next line which is the original comment line).
            # However we want to replace the comment line with our own comment? We want to keep the comment line after our variables.
            # So we will add our variable lines, then not add the original line? Actually we already added the '<script>' line; the next line in original is the comment line.
            # We'll add our variable lines, then let the loop continue and add the original comment line.
            # But we need to adjust indentation: the original comment line has 8 spaces then '// Feature Importance Bar Chart'.
            # We'll add our variable lines with same indentation.
            # Let's compute indent from the next line.
            if i+1 < len(lines):
                next_line = lines[i+1]
                # match leading spaces
                match = re.match(r'(\s*)', next_line)
                if match:
                    indent = match.group(1)
                else:
                    indent = ''
            else:
                indent = ''

            # Add variable lines
            for v in [
                '// Chart data from Flask',
                'const feature_names = {{ feature_names_json|safe }};',
                'const feature_importances = {{ feature_importances|safe }};',
                'const fake_count = {{ fake_count|safe }};',
                'const real_count = {{ real_count|safe }};',
                'const prediction_percentages = {{ prediction_percentages|safe }};',
                'const engagement_bins = {{ engagement_bins|safe }};',
                'const engagement_counts = {{ engagement_counts|safe }};',
                '// Feature Importance Bar Chart'
            ]:
                new_lines.append(indent + v)
            # Skip adding the original comment line because we already added it above.
            i += 1  # skip the original comment line
            continue
        i += 1

    # Join back
    new_content = ''.join(new_lines)

    # Now we need to replace the data in the feature chart to use the variables.
    # We'll replace the lines:
    #                 labels: ['Followers', 'Following', 'Posts', 'Engagement Rate (%)', 'Avg Likes per Post', 'Avg Comments per Post', 'Verified', 'Account Age (Years)', 'Bio Text'],
    #                 datasets: [{
    #                     label: 'Importance Score',
    #                     data: [0.18, 0.12, 0.15, 0.22, 0.10, 0.08, 0.05, 0.05, 0.05],
    # We'll replace with:
    #                 labels: feature_names,
    #                 datasets: [{
    #                     label: 'Importance Score',
    #                     data: feature_importances,
    # We'll also replace backgroundColor and borderColor to generate arrays of same length as feature_names.
    # We'll replace the whole backgroundColor and borderColor arrays with something like:
    #                     backgroundColor: feature_names.map((_, i) => `rgba(${(i*40)%255}, ${(i*80+100)%255}, ${(i*120+150)%255}, 0.8)`),
    #                     borderColor: feature_names.map((_, i) => `rgba(${(i*40)%255}, ${(i*80+100)%255}, ${(i*120+150)%255}, 1)`),
    # But that's complex; we can just set a single color for all.
    # Let's do a simple approach: set backgroundColor: 'rgba(110, 142, 251, 0.8)' and borderColor: 'rgba(110,142,251,1)' (single strings) - Chart.js will apply to all bars.
    # According to Chart.js, if you provide a string, it's used for all points.
    # We'll do that.

    # We'll do regex replacement for the data object inside the featureChart.
    # We'll locate the featureChart block and replace the data property.
    # Instead of complex regex, we can do string replacement for the specific lines we know.
    # Let's do multiple replacements.

    # Replace labels line
    new_content = re.sub(r"""labels: \['Followers', 'Following', 'Posts', 'Engagement Rate \(%\)', 'Avg Likes per Post', 'Avg Comments per Post', 'Verified', 'Account Age \(Years\)', 'Bio Tech'""",
                         "labels: feature_names", new_content)
    # Oops we need exact match. Let's just replace the whole line with regex that matches the line.
    # We'll do line by line again.

    lines = new_content.splitlines(keepends=True)
    out_lines = []
    for line in lines:
        if line.strip().startswith('labels: [') and 'Followers' in line:
            # replace with
            indent = len(line) - len(line.lstrip())
            out_lines.append(' ' * indent + 'labels: feature_names,')
        elif line.strip().startswith('data: [') and '0.18' in line:
            indent = len(line) - len(line.lstrip())
            out_lines.append(' ' * indent + 'data: feature_importances,')
        elif line.strip().startswith('backgroundColor: ['):
            # replace with single color
            indent = len(line) - len(line.lstrip())
            out_lines.append(' ' * indent + "backgroundColor: 'rgba(110, 142, 251, 0.8)',")
        elif line.strip().startswith('borderColor: ['):
            indent = len(line) - len(line.lstrip())
            out_lines.append(' ' * indent + "borderColor: 'rgba(110, 142, 251, 1)',")
        else:
            out_lines.append(line)
    new_content = ''.join(out_lines)

    # Now replace the pie chart data to use fake_count and real_count.
    # Find the pieChart data line:
    #                 data: [342, 898],
    # Replace with:
    #                 data: [fake_count, real_count],
    lines = new_content.splitlines(keepends=True)
    out_lines = []
    for line in lines:
        if line.strip().startswith('data: [') and '342' in line:
            indent = len(line) - len(line.lstrip())
            out_lines.append(' ' * indent + f'data: [fake_count, real_count],')
        else:
            out_lines.append(line)
    new_content = ''.join(out_lines)

    # Also we might want to keep the backgroundColor and borderColor for pie chart; they are fine.

    # Write back
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print('Updated analytics.html')

if __name__ == '__main__':
    main()