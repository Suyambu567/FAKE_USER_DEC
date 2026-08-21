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
            indent = len(line) - len(line.lstrip())
            out.append(' ' * indent + 'labels: feature_names,\n')
            i += 1
            continue
        # Replace data line
        if stripped.startswith('data: [') and '0.18' in stripped:
            indent = len(line) - len(line.lstrip())
            out.append(' ' * indent + 'data: feature_importances,\n')
            i += 1
            continue
        # Replace backgroundColor array
        if stripped.startswith('backgroundColor: ['):
            indent = len(line) - len(line.lstrip())
            out.append(' ' * indent + "backgroundColor: 'rgba(110, 142, 251, 0.8)',\n")
            i += 1
            # Skip until we pass the closing bracket line
            while i < len(lines):
                if lines[i].lstrip().startswith(']'):
                    i += 1  # skip the closing bracket line
                    break
                i += 1
            continue
        # Replace borderColor array
        if stripped.startswith('borderColor: ['):
            indent = len(line) - len(line.lstrip())
            out.append(' ' * indent + "borderColor: 'rgba(110, 142, 251, 1)',\n")
            i += 1
            while i < len(lines):
                if lines[i].lstrip().startswith(']'):
                    i += 1
                    break
                i += 1
            continue
        # Default: keep line
        out.append(line)
        i += 1

    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(out)
    print('Updated chart data lines.')

if __name__ == '__main__':
    main()