"""Merge lossless NPZ parts using only the Python standard library."""
from pathlib import Path
import hashlib,json,os

def main():
    root=Path(__file__).resolve().parent
    meta=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    output=root/meta['output_file'];temporary=root/(meta['output_file']+'.merging')
    if output.exists():raise FileExistsError('Output already exists; preserved: '+str(output))
    if temporary.exists():raise FileExistsError('Partial merge already exists; inspect it first: '+str(temporary))
    # Check every volume before creating the output.
    for item in meta['parts']:
        p=root/item['file']
        if not p.is_file():raise FileNotFoundError('Missing part: '+p.name)
        if p.stat().st_size!=item['bytes']:raise ValueError('Incorrect part size: '+p.name)
        with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest!=item['sha256']:raise ValueError('Part checksum failed: '+p.name)
    total=hashlib.sha256();size=0
    with temporary.open('xb') as out:
        for item in meta['parts']:
            with (root/item['file']).open('rb') as f:
                while block:=f.read(4*1024*1024):
                    out.write(block);total.update(block);size+=len(block)
            print('Merged '+item['file'],flush=True)
        out.flush();os.fsync(out.fileno())
    if size!=meta['total_bytes'] or total.hexdigest()!=meta['sha256']:
        raise ValueError('Merged checksum failed; partial file retained for inspection')
    temporary.rename(output)
    print('SUCCESS: '+str(output))
    print('SHA256: '+total.hexdigest())

if __name__=='__main__':main()
