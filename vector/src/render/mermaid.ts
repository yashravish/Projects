import { spawn } from 'node:child_process';

export async function renderMermaid(inputPath: string, outPath: string, format: 'png' | 'svg'): Promise<void> {
  return new Promise((resolve, reject) => {
    const proc = spawn('mmdc', ['-i', inputPath, '-o', outPath, '-f', format], { stdio: 'ignore' });
    proc.on('error', (err) => {
      reject(new Error('Mermaid CLI (mmdc) not found. Install: npm i -g @mermaid-js/mermaid-cli'));
    });
    proc.on('exit', (code) => {
      if (code === 0) resolve(); else reject(new Error(`mmdc exited with code ${code}`));
    });
  });
}


