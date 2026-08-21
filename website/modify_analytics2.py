import re

def main():
    with open('templates/analytics.html', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find indices of section comments
    comments = []
    for i, line in enumerate(lines):
        if '<!--' in line and ('Feature Importance' in line or 'Prediction Distribution' in line or 'System Health Status' in line):
            comments.append(i)
    if len(comments) != 3:
        print(f"Expected 3 comments, found {len(comments)}")
        return

    # For each comment, find the start of the div.content-section (next line that starts with spaces and <div)
    def find_section_start(comment_idx):
        i = comment_idx + 1
        while i < len(lines):
            if lines[i].strip().startswith('<div class="content-section">'):
                return i
            i += 1
        return None

    # Find the end of the section: the matching closing </div> (assuming same indentation)
    def find_section_end(start_idx):
        # Assume the section ends at the next line that starts with exactly the same indentation as the opening div and is '</div>'
        # The opening div line:
        open_line = lines[start_idx]
        # Determine indentation: count leading spaces
        indent = len(open_line) - len(open_line.lstrip())
        i = start_idx + 1
        while i < len(lines):
            line = lines[i]
            if line.strip() == '</div>' and (len(line) - len(line.lstrip())) == indent:
                return i  # this is the closing div line
            i += 1
        return None

    starts = []
    ends = []
    for c in comments:
        s = find_section_start(c)
        if s is None:
            print(f"Could not find start for comment at line {c}")
            return
        e = find_section_end(s)
        if e is None:
            print(f"Could not find end for section starting at {s}")
            return
        starts.append(s)
        ends.append(e)

    # Now we have sections 0,1,2 with start/end lines inclusive.
    # We want to keep sections 0 and 1, wrap them in a div class="charts-row".
    # Remove section 2 entirely.

    # Build new lines:
    new_lines = []
    i = 0
    while i < len(lines):
        if i == starts[0]:
            # start of first section: insert opening wrapper div
            new_lines.append('        <div class="charts-row">\n')
            # copy from start of first section to end of second section
            # we'll copy sections 0 and 1, including any whitespace between them.
            # We'll copy lines from starts[0] to ends[1] inclusive.
            for j in range(starts[0], ends[1] + 1):
                new_lines.append(lines[j])
            # after second section, close wrapper div
            new_lines.append('        </div>\n')
            # skip to after second section
            i = ends[1] + 1
            # Now we need to skip the third section entirely.
            # Move i to after the third section's end.
            i = ends[2] + 1
            continue
        # If we are inside the third section, skip until its end.
        if starts[2] <= i <= ends[2]:
            i = ends[2] + 1
            continue
        new_lines.append(lines[i])
        i += 1

    # Now we need to add CSS for .charts-row before the closing </style> tag.
    html = ''.join(new_lines)
    # Insert CSS before </style>
    css = '''\n/* Chart rows layout */\n.charts-row {\n    display: flex;\n    gap: 1.5rem;\n    margin: 0 -2rem 2rem -2rem;\n    flex-wrap: wrap;\n}\n.charts-row > .content-section {\n    flex: 1 1 45%;\n    min-width: 250px;\n    margin: 0;\n}\n@media (max-width: 900px) {\n    .charts-row > .content-section {\n        flex: 1 1 100%;\n    }\n}\n'''
    # Find the </style> tag
    style_end = html.find('</style>')
    if style_end == -1:
        print("Could not find </style>")
        return
    html = html[:style_end] + css + '\n' + html[style_end:]

    # Ensure canvas for featureImportanceChart has height:400px;width:100%
    html = re.sub(r'(<canvas id="featureImportanceChart")[^>]*>', r'\1 style="height:400px;width:100%;">', html)

    with open('templates/analytics.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print("Done")

if __name__ == '__main__':
    main()