"""Stop only the retained native Inkling container below 12 GiB host headroom."""
import json
import os
from pathlib import Path
import subprocess
import time

root = Path(os.environ.get('INKLING_NATIVE_THINKING_ROOT', '/home/kostadis/inkling-native-thinking-20261122'))
identity = json.loads((root / 'before.json').read_text())['native']['id']
record = {'started_unix': time.time(), 'samples': 0, 'floor_tripped': False,
          'min_host_available_bytes': None, 'max_swap_used_bytes': 0}
while True:
    container = json.loads(subprocess.check_output(['docker', 'inspect', 'inkling2mini'], text=True))[0]
    assert container['Id'] == identity
    if not container['State']['Running']:
        break
    memory = {line.split(':')[0]: int(line.split()[1]) * 1024
              for line in Path('/proc/meminfo').read_text().splitlines()}
    free = memory['MemAvailable']; record['samples'] += 1
    record['min_host_available_bytes'] = min(free, record['min_host_available_bytes'] or free)
    record['last_host_available_bytes'] = free; record['updated_unix'] = time.time()
    record['max_swap_used_bytes'] = max(record['max_swap_used_bytes'], memory['SwapTotal'] - memory['SwapFree'])
    if free < 12 * 1024**3:
        record['floor_tripped'] = True
        subprocess.run(['docker', 'stop', '--timeout', '30', 'inkling2mini'], check=True)
    temporary = root / 'memory.tmp'; temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(root / 'memory.json')
    if record['floor_tripped']:
        break
    time.sleep(2)
