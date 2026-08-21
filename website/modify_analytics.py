import re

with open('templates/analytics.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the system health chart section (from its comment to its closing </div> of the content-section)
# We'll remove from the line containing '<!-- System Health Status (Additional Chart - Bar Chart for Key Metrics) -->'
# up to and including the closing </div> that matches the opening of that content-section.
# Simpler: replace the block between that comment and the next '</div>\n        </div>\n    </div>' pattern.
# We'll do a regex that captures from the comment to the closing </div> of that section, but not including the outer wrapper.
# Let's just split and rejoin.

# Find start index
start_marker = '<!-- System Health Status (Additional Chart - Bar Chart for Key Metrics) -->'
start_idx = content.find(start_marker)
if start_idx == -1:
    print("Start marker not found")
    exit(1)

# Find the corresponding end of that content-section.
# We'll iterate forward counting opening and closing div tags for div class="content-section".
# But we know it's the third content-section inside the charts-row.
# Let's just find the next '</div>' that is at the same indentation level as the opening <div class="content-section">.
# Simpler: remove from start_idx to the position of the next '</div>\n        </div>\n    </div>' after the chart container.
# We'll search for the pattern: '</div>\n\s*</div>\n\s*</div>' after the canvas line.

# Let's just do a more robust approach: find the index of the closing </div> of the charts-row wrapper after the three sections.
# We'll instead rebuild: keep everything before start_idx, then skip until after the closing </div> of that section, then continue.

# We'll find the end of that content-section by scanning for balanced braces of '<div class="content-section">' and '</div>' starting at start_idx.
# Actually the opening <div class="content-section"> is a few lines before start_idx? Let's locate it.
# Look backwards for '<div class="content-section">'
open_div = content.rfind('<div class="content-section">', 0, start_idx)
if open_div == -1:
    print("Opening div not found")
    exit(1)
# Now from open_div, find the matching closing div.
depth = 0
i = open_div
while i < len(content):
    if content[i:i+len('<div class="content-section">')] == '<div class="content-section">':
        depth += 1
        i += len('<div class="content-section">')
        continue
    if content[i:i+6] == '</div>':
        depth -= 1
        if depth == 0:
            close_idx = i + 6  # position after the closing '</div>'
            break
        i += 6
        continue
    i += 1
else:
    print("Could not find matching closing div")
    exit(1)

# Now we have the range [open_dev, close_idx) to remove.
new_content = content[:open_div] + content[close_idx:]

# Also need to remove the corresponding JavaScript for the health chart.
# Find the section for healthChart in the script.
# We'll remove from the line containing '// System Health Status - Key Metrics Bar Chart' to the closing '});' of that chart init.
# Simpler: we can delete the block between that comment and the next '});' that is at same indentation level? We'll just remove the whole block for healthChart.
# Let's find the start of that comment.
js_start = new_content.find('// System Health Status - Key Metrics Bar Chart')
if js_start != -1:
    # Find the end of that block: look for the closing '});' of the Chart constructor.
    # We'll find the next '});' after js_start that is at same brace level? We'll just find the next '});' and assume it's the end.
    # But there may be nested braces; however the chart options are simple.
    # We'll find the position of the next '});' after js_start.
    brace_end = new_content.find('});', js_start)
    if brace_end == -1:
        print("Could not find end of health chart JS")
    else:
        # Include the '});' itself.
        js_end = brace_end + 3
        # Remove from js_start to js_end
        new_content = new_content[:js_start] + new_content[js_end:]

# Write back
with open('templates/analytics.html', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Modified analytics.html")
