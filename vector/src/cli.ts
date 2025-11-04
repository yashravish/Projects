import { Command } from 'commander';
import path from 'node:path';
import fs from 'node:fs/promises';
import { loadConfig, writeDefaultConfig, ensureProjectDirs, type VDiagramConfig, resolveConfigPath } from './config.js';
import { createLogger } from './log.js';

async function main(argv: string[]) {
  const program = new Command();
  program
    .name('vdiagram')
    .description('Vector DB → Mermaid diagram CLI (offline-first)')
    .version('0.1.0')
    .option('--config <path>', 'Path to config file')
    .option('--verbose', 'Enable verbose logging', false)
    .option('--quiet', 'Reduce logging output', false)
    .option('--offline', 'Force offline mode (disallow network)', false)
    .option('--no-color', 'Disable colored output', false);

  program
    .command('init')
    .description('Initialize project config and local directories')
    .option('--allow-network', 'Permit network to download model on first run', false)
    .action(async (opts) => {
      const parent = program.opts();
      const logger = createLogger({ verbose: parent.verbose, quiet: parent.quiet, color: parent.color });
      const configPath = await resolveConfigPath(parent.config);
      const cfg = await writeDefaultConfig(configPath);
      await ensureProjectDirs(cfg);
      logger.info(`Initialized config at ${configPath}`);
      if (opts.allow_network) {
        logger.info('Network allowed: you can run `vdiagram models pull` to pre-download the embedding model.');
      } else {
        logger.info('Offline mode: model will not be downloaded automatically. Use `vdiagram models pull --allow-network` later.');
      }
    });

  const models = program.command('models').description('Manage local models (offline cache)');
  models
    .command('pull')
    .description('Pre-download embedding model to local cache')
    .option('--allow-network', 'Permit network access to fetch model', false)
    .action(async (opts) => {
      const parent = program.opts();
      const logger = createLogger({ verbose: parent.verbose, quiet: parent.quiet, color: parent.color });
      const configPath = await resolveConfigPath(parent.config);
      const cfg = await loadConfig(configPath);
      await ensureProjectDirs(cfg);
      const allowNet = !!opts.allow_network || !parent.offline;
      if (!allowNet) {
        logger.warn('Offline mode: cannot download model. Re-run with: vdiagram models pull --allow-network');
        logger.info(`Expected local model path: ${cfg.modelPath}`);
        return;
      }
      logger.info('Preparing to download model using @xenova/transformers...');
      try {
        const cacheDir = cfg.modelPath;
        await fs.mkdir(cacheDir, { recursive: true });
        process.env.TRANSFORMERS_CACHE = cacheDir;
        // Lazy import so regular CLI usage doesn’t require the package until needed
        const { pipeline } = await import('@xenova/transformers');
        // Trigger first-run download
        const pipe = await pipeline('feature-extraction', `Xenova/${cfg.embeddingModel}`);
        await pipe('hello');
        logger.info(`Model cached at: ${cacheDir}`);
      } catch (err: any) {
        logger.error(`Failed to pull model: ${err?.message ?? String(err)}`);
        process.exitCode = 1;
      }
    });

  program
    .command('ingest')
    .argument('<paths...>', 'Paths to ingest')
    .option('--glob <pattern>', 'Glob pattern to include')
    .option('--ignore <pattern>', 'Glob pattern to ignore (repeatable)')
    .option('--tag <tag>', 'Tag to assign to ingested documents')
    .option('--reset', 'Reset store for provided tag before ingest', false)
    .option('--batch-size <n>', 'Embedding batch size', (v: string) => parseInt(v, 10), 128)
    .option('--concurrency <n>', 'Embedding concurrency', (v: string) => parseInt(v, 10), 4)
    .option('--max-file-size <bytes>', 'Max file size to ingest', (v: string) => parseInt(v, 10), 1048576)
    .option('--target-chunk-size <n>', 'Target chunk size (chars)', (v: string) => parseInt(v, 10), 900)
    .option('--overlap <n>', 'Chunk overlap (chars)', (v: string) => parseInt(v, 10), 140)
    .option('--max-chunk-size <n>', 'Max chunk size (chars)', (v: string) => parseInt(v, 10), 1600)
    .option('--allow-network', 'Permit network to fetch model if missing', false)
    .action(async (paths: string[], opts: { glob?: string; ignore?: string; tag?: string; reset?: boolean; maxFileSize: number; targetChunkSize: number; overlap: number; maxChunkSize: number; allowNetwork?: boolean; }) => {
      const parent = program.opts();
      const logger = createLogger({ verbose: parent.verbose, quiet: parent.quiet, color: parent.color });
      const configPath = await resolveConfigPath(parent.config);
      const cfg = await loadConfig(configPath);
      const { ingestEmbedUpsert } = await import('./ingest/ingest_and_upsert.js');
      const includeGlobs = opts.glob ? [opts.glob] : undefined;
      const ignoreGlobs = opts.ignore ? [opts.ignore] : undefined;
      const { stats, upserted } = await ingestEmbedUpsert(paths, cfg, logger, {
        includeGlobs,
        ignoreGlobs,
        maxFileSizeBytes: opts.maxFileSize,
        targetChunkSize: opts.targetChunkSize,
        overlap: opts.overlap,
        maxChunkSize: opts.maxChunkSize,
        tag: opts.tag,
        reset: opts.reset,
        allowNetwork: !!opts.allowNetwork && !parent.offline,
      });
      logger.info(`Ingest summary → files: ${stats.chunkedFiles}, chunks: ${stats.chunks}, upserted: ${upserted}, skipped: ${stats.skippedUnchanged}`);
    });

  const diagram = program.command('diagram').description('Generate diagrams from local context');
  function diagramCommon(cmd: Command) {
    return cmd
      .option('--source <path|tag>', 'Source restriction (path or tag)')
      .option('--k <n>', 'Retrieval depth', (v) => parseInt(v, 10), 24)
      .option('--out <file>', 'Output file for diagram text')
      .option('--format <fmt>', 'Diagram format (mermaid|plantuml|dot)', 'mermaid')
      .option('--render <fmt>', 'Render output (png|svg)')
      .option('--allow-network', 'Permit network access for renderers', false);
  }
  diagramCommon(
    diagram
      .command('sequence')
      .requiredOption('--question <text>', 'Prompt/question for sequence diagram')
  ).action(async (opts) => {
    const parent = program.opts();
    const logger = createLogger({ verbose: parent.verbose, quiet: parent.quiet, color: parent.color });
    const configPath = await resolveConfigPath(parent.config);
    const cfg = await loadConfig(configPath);
    const { openVectorStore } = await import('./store/index.js');
    const { createEmbedder } = await import('./embed/embeddings.js');
    const { retrieve } = await import('./retrieval/retriever.js');
    const { generateSequenceMermaid } = await import('./generate/sequence.js');
    const { validateMermaid } = await import('./validate/mermaid.js');
    const { saveDiagram } = await import('./diagrams/store.js');
    const store = await openVectorStore(cfg.dbPath);
    const embedder = await createEmbedder(cfg, !!opts.allowNetwork && !parent.offline);
    const [qVec] = await embedder.embedTexts([opts.question]);
    const results = await retrieve(store, qVec, { k: opts.k ?? cfg.defaultK, lambda: cfg.mmrLambda, minScore: cfg.minScore, filterPathPrefix: opts.source });
    const { dsl, provenance } = generateSequenceMermaid(opts.question, results);
    const val = validateMermaid(dsl);
    if (!val.ok) { logger.error(`Invalid diagram: ${val.error}`); process.exitCode = 1; return; }
    const rec = await saveDiagram(cfg.indexPath, { type: 'sequence', dsl, params: { k: opts.k }, sources: provenance, outPath: opts.out });
    logger.info(`Diagram saved to ${rec.path}`);
  });

  diagramCommon(
    diagram
      .command('class')
      .requiredOption('--topic <text>', 'Topic/module/entity for class diagram')
  ).action(async (opts) => {
    const parent = program.opts();
    const logger = createLogger({ verbose: parent.verbose, quiet: parent.quiet, color: parent.color });
    const configPath = await resolveConfigPath(parent.config);
    const cfg = await loadConfig(configPath);
    const { openVectorStore } = await import('./store/index.js');
    const { createEmbedder } = await import('./embed/embeddings.js');
    const { retrieve } = await import('./retrieval/retriever.js');
    const { generateClassMermaid } = await import('./generate/class.js');
    const { validateMermaid } = await import('./validate/mermaid.js');
    const { saveDiagram } = await import('./diagrams/store.js');
    const store = await openVectorStore(cfg.dbPath);
    const embedder = await createEmbedder(cfg, !!opts.allowNetwork && !parent.offline);
    const [qVec] = await embedder.embedTexts([opts.topic]);
    const results = await retrieve(store, qVec, { k: opts.k ?? cfg.defaultK, lambda: cfg.mmrLambda, minScore: cfg.minScore, filterPathPrefix: opts.source });
    const { dsl, provenance } = generateClassMermaid(opts.topic, results);
    const val = validateMermaid(dsl);
    if (!val.ok) { logger.error(`Invalid diagram: ${val.error}`); process.exitCode = 1; return; }
    const rec = await saveDiagram(cfg.indexPath, { type: 'class', dsl, params: { k: opts.k }, sources: provenance, outPath: opts.out });
    logger.info(`Diagram saved to ${rec.path}`);
  });

  diagramCommon(
    diagram
      .command('flow')
      .requiredOption('--question <text>', 'Prompt/question for flow diagram')
  ).action(async (opts) => {
    const parent = program.opts();
    const logger = createLogger({ verbose: parent.verbose, quiet: parent.quiet, color: parent.color });
    const configPath = await resolveConfigPath(parent.config);
    const cfg = await loadConfig(configPath);
    const { openVectorStore } = await import('./store/index.js');
    const { createEmbedder } = await import('./embed/embeddings.js');
    const { retrieve } = await import('./retrieval/retriever.js');
    const { generateFlowMermaid } = await import('./generate/flow.js');
    const { validateMermaid } = await import('./validate/mermaid.js');
    const { saveDiagram } = await import('./diagrams/store.js');
    const store = await openVectorStore(cfg.dbPath);
    const embedder = await createEmbedder(cfg, !!opts.allowNetwork && !parent.offline);
    const [qVec] = await embedder.embedTexts([opts.question]);
    const results = await retrieve(store, qVec, { k: opts.k ?? cfg.defaultK, lambda: cfg.mmrLambda, minScore: cfg.minScore, filterPathPrefix: opts.source });
    const { dsl, provenance } = generateFlowMermaid(opts.question, results);
    const val = validateMermaid(dsl);
    if (!val.ok) { logger.error(`Invalid diagram: ${val.error}`); process.exitCode = 1; return; }
    const rec = await saveDiagram(cfg.indexPath, { type: 'flow', dsl, params: { k: opts.k }, sources: provenance, outPath: opts.out });
    logger.info(`Diagram saved to ${rec.path}`);
  });

  program
    .command('show')
    .argument('<idOrFile>', 'Diagram id or file')
    .action(async (idOrFile: string) => {
      const parent = program.opts();
      const logger = createLogger({ verbose: parent.verbose, quiet: parent.quiet, color: parent.color });
      const { getDiagramById } = await import('./diagrams/store.js');
      const { resolveConfigPath, loadConfig } = await import('./config.js');
      const configPath = await resolveConfigPath(parent.config);
      const cfg = await loadConfig(configPath);
      let file = idOrFile;
      if (!/\.mmd$/i.test(idOrFile)) {
        const meta = await getDiagramById(cfg.indexPath, idOrFile);
        if (!meta) { logger.error('Diagram not found'); process.exitCode = 1; return; }
        file = meta.path;
      }
      const fs = await import('node:fs/promises');
      const dsl = await fs.readFile(file, 'utf8');
      // eslint-disable-next-line no-console
      console.log(dsl);
    });

  program
    .command('export')
    .requiredOption('--format <fmt>', 'mermaid|plantuml|dot')
    .requiredOption('--out <file>', 'Output file path')
    .argument('<idOrFile>', 'Diagram id or file')
    .action(async (idOrFile: string, opts: { format: string; out: string }) => {
      const parent = program.opts();
      const logger = createLogger({ verbose: parent.verbose, quiet: parent.quiet, color: parent.color });
      const { getDiagramById } = await import('./diagrams/store.js');
      const { resolveConfigPath, loadConfig } = await import('./config.js');
      const configPath = await resolveConfigPath(parent.config);
      const cfg = await loadConfig(configPath);
      let file = idOrFile;
      if (!/\.mmd$/i.test(idOrFile)) {
        const meta = await getDiagramById(cfg.indexPath, idOrFile);
        if (!meta) { logger.error('Diagram not found'); process.exitCode = 1; return; }
        file = meta.path;
      }
      if (opts.format === 'mermaid') {
        const fs = await import('node:fs/promises');
        const dsl = await fs.readFile(file, 'utf8');
        await fs.writeFile(opts.out, dsl, 'utf8');
        logger.info(`Exported Mermaid to ${opts.out}`);
      } else {
        logger.warn('Only Mermaid pass-through export supported in MVP.');
      }
    });

  program
    .command('stats')
    .action(async () => {
      const parent = program.opts();
      const logger = createLogger({ verbose: parent.verbose, quiet: parent.quiet, color: parent.color });
      logger.info('Stats is not yet implemented. This is a stub.');
    });

  program
    .command('reindex')
    .action(async () => {
      const parent = program.opts();
      const logger = createLogger({ verbose: parent.verbose, quiet: parent.quiet, color: parent.color });
      logger.info('Reindex is not yet implemented. This is a stub.');
    });

  program
    .command('purge')
    .option('--tag <t>', 'Purge only a specific tag')
    .option('--all', 'Purge entire index', false)
    .action(async () => {
      const parent = program.opts();
      const logger = createLogger({ verbose: parent.verbose, quiet: parent.quiet, color: parent.color });
      logger.info('Purge is not yet implemented. This is a stub.');
    });

  await program.parseAsync(argv);
}

main(process.argv).catch((err) => {
  // eslint-disable-next-line no-console
  console.error(err);
  process.exit(1);
});


