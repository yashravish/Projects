import * as kleur from 'kleur';

type LoggerOptions = {
  verbose?: boolean;
  quiet?: boolean;
  color?: boolean;
};

export function createLogger(opts: LoggerOptions) {
  const useColor = opts.color !== false;
  const c = useColor ? kleur : new Proxy(kleur, {
    get: () => (x: any) => String(x),
  });
  function log(level: 'info' | 'warn' | 'error' | 'debug', msg: string) {
    if (opts.quiet && level === 'info') return;
    if (!opts.verbose && level === 'debug') return;
    const prefix = level === 'info' ? c.blue('[info]')
      : level === 'warn' ? c.yellow('[warn]')
      : level === 'error' ? c.red('[error]')
      : c.magenta('[debug]');
    // eslint-disable-next-line no-console
    console.log(prefix, msg);
  }
  return {
    info: (m: string) => log('info', m),
    warn: (m: string) => log('warn', m),
    error: (m: string) => log('error', m),
    debug: (m: string) => log('debug', m),
  };
}


