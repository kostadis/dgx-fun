"""Guard only the selected Inkling container while leaving Clef untouched."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--container', required=True)
    p.add_argument('--profile', required=True)
    a = p.parse_args()
    result = {'container_id': a.container, 'floor_bytes': 12 * 1024**3,
              'floor_tripped': False, 'samples': 0, 'min_available_bytes': None,
              'max_swap_used_bytes': 0, 'started_unix': time.time()}
    output = a.root / (a.profile + '-memory.json')
    while True:
        d = json.loads(subprocess.check_output(['docker', 'inspect', a.container], text=True))[0]
        if not d['State']['Running']:
            break
        memory = {line.split(':')[0]: int(line.split()[1]) * 1024
                  for line in Path('/proc/meminfo').read_text().splitlines()}
        available = memory['MemAvailable']
        result['samples'] += 1
        result['updated_unix'] = time.time()
        result['last_available_bytes'] = available
        result['min_available_bytes'] = min(available, result['min_available_bytes'] or available)
        result['max_swap_used_bytes'] = max(result['max_swap_used_bytes'],
                                            memory['SwapTotal'] - memory['SwapFree'])
        if available < result['floor_bytes']:
            result['floor_tripped'] = True
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(output)
        if result['floor_tripped']:
            subprocess.run(['docker', 'stop', '--timeout', '15', a.container], check=True)
            break
        time.sleep(2)


if __name__ == '__main__':
    main()
