import re

def main():
    path = 'templates/analytics.html'
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        # Replace labels line
        if stripped.startswith('labels: [') and 'Followers' in stripped:
            # Keep same indentation
            indent = len(line) - len(linediff := line.lstrip())
            out.append(' ' * indent + 'labels: feature_names,')
            i += 1
            continue
        # Replace data line
        if stripped.startswith('data: [') and '0.18' in stripped:
            indent = len(line) - len(linediff := line.lstrip())
            out.append(' ' * indent + 'data: feature_importances,')
            i += 1
            continue
        # Replace backgroundColor array (multiple lines)
        if stripped.startswith('backgroundColor: ['):
            # Skip until we find the closing line that ends with '],'
            indent = len(line) - len(linediff := line.lstrip())
            # Output replacement line
            out.append(' ' * indent + "backgroundColor: 'rgba(110, 142, 251, 0.8)',")
            # Now skip lines until we pass the closing bracket line
            i += 1
            while i < len(lines):
                if lines[i].lstrip().startswith(']'):
                    # Skip this line
                    i += 1
                    break
                i += 1
            continue
        # Replace borderColor array
        if stripped.startswith('borderColor: ['):
            indent = len(line) - len(linediff := line.lstrip())
            out.append(' ' * indent + "borderColor: 'rgba(110, 142, 251, 1)',")
            i += 1
            while i < len(lines):
                if lines[i].lstrip().startswith(']'):
                    i += 1
                    break
                i += 1
            continue
        # If none matched, keep line
        out.append(line)
        i += 1

    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(out)
    print('Updated chart data lines.')

if __name__ == '__main__':
    main()