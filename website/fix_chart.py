import re

def fix_feature_chart(content):
    # Find the featureChart block
    # We'll locate the line with 'const featureChart = new Chart(featureCtx, {'
    lines = content.splitlines(keepends=True)
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if 'const featureChart = new Chart(featureCtx, {' in line:
            # We have found the start. We'll keep this line and the next lines until we hit the closing brace of the chart options?
            # Instead, we will replace from this line until the line that contains '});' that ends this chart call (but careful there are nested braces).
            # Simpler: we will just replace the data object inside.
            # We'll output the line as is, then continue copying lines until we reach the 'data: {' line, then we will replace until the matching '}' (balanced).
            out.append(line)
            i += 1
            # Copy lines until we see 'data: {'
            while i < len(lines) and not lines[i].strip().startswith('data: {'):
                out.append(lines[i])
                i += 1
            if i >= len(lines):
                break
            # Now we are at the line with 'data: {'
            # We'll replace from this line until the matching closing brace (considering nested braces).
            # Let's find the matching brace.
            start_idx = i
            brace_count = 0
            while i < len(lines):
                l = lines[i]
                for ch in l:
                    if ch == '{':
                        brace_count += 1
                    elif ch == '}':
                        brace_count -= 1
                out.append(l)
                i += 1
                if brace_count == 0:
                    break
            # Now we have replaced the original data block with whatever we appended (which is the original). We need to replace it with new content.
            # Instead of the above complex approach, let's just do a regex replacement on the whole content.
            break
        else:
            out.append(line)
            i += 1
    # If we broke out, we need to reconstruct before and after.
    # Let's just do a regex substitution on the whole content for simplicity.
    # We'll replace the pattern:
    # data: {
    #     labels: \[.*?\],
    #     datasets: \[
    #         {
    #             label: 'Importance Score',
    #             data: \[.*?\],
    #             backgroundColor: \[.*?\],
    #             borderColor: \[.*?\],
    #             borderWidth: \d+
    #         }
    #     ]
    # }
    # with new version.
    # However regex across lines is tricky with nested brackets.
    # Given time, we'll do a simpler approach: replace the lines we know are wrong by re-reading the file and writing corrected lines.
    return None

if __name__ == '__main__':
    with open('templates/analytics.html', 'r', encoding='utf-8') as f:
        content = f.read()
    # We'll just replace the problematic lines using multiple regexes.
    # Fix the labels line: ensure there is a newline after the comma.
    # Find pattern: 'labels: [^,]*,' and ensure newline after.
    # We'll do line-by-line processing again but smarter.
    lines = content.splitlines(keepends=True)
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith('labels:') and 'feature_names' in stripped:
            # Ensure proper formatting: we want "                    labels: feature_names,"
            # Keep leading spaces.
            indent = len(line) - len(lined)
            # Actually we need to compute leading spaces of original line.
            indent = len(line) - len(line.lstrip())
            # Ensure there is exactly one space after colon? We'll just set:
            new_line = ' ' * indent + 'labels: feature_names,'
            # Ensure no extra stuff after.
            out.append(new_line + '\n')
            i += 1
            continue
        if stripped.startswith('data:') and 'feature_importances' in stripped:
            indent = len(line) - len(line.lstrip())
            new_line = ' ' * indent + 'data: feature_importances,'
            out.append(new_line + '\n')
            i += 1
            continue
        if stripped.startswith('backgroundColor:') and \"'rgba(110, 142, 251, 0.8)'\" in line:
            # We have duplicate lines; we want to keep only one line.
            # We'll skip until we see a line that is not part of the backgroundColor array.
            # Let's just skip this line and continue; we will later add the correct line.
            # But we need to also skip the following lines that are part of the old array.
            # We'll advance until we see a line that starts with same indent and contains 'borderColor:' maybe.
            # We'll just skip this line and not add anything; we'll add the correct line later when we encounter the proper place.
            i += 1
            # We'll look ahead to skip lines that are part of the old array.
            while i < len(lines) and lines[i].lstrip().startswith("'"):
                i += 1
            # Now we are at the line after the old array (should be the line with closing bracket of backgroundColor? Actually the array ended with a line containing just '],'?)
            # We'll just continue; we will add the correct line when we encounter the placeholder? Not good.
            # Let's instead just replace the whole block from backgroundColor line to the line before borderColor? Too messy.
            # We'll instead do a full rewrite of the data block later.
            pass
        else:
            out.append(line)
            i += 1
    # This is getting too complex.
    # Let's instead write a new version of the file by copying the original and using sed commands via bash.
    # We'll exit and do that.
    print("Done")