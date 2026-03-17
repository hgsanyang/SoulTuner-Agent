import { z } from 'zod';
import { config } from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

// Define ES module equivalents
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Try multiple .env locations: load ALL found (later files don't overwrite earlier values)
const candidatePaths = [
  path.resolve(__dirname, '../../../../.env'),   // dev: src/config/ → Muisc-Research/.env
  path.resolve(__dirname, '../../../../../.env'), // dist: dist/config/ → Muisc-Research/.env
  path.resolve(__dirname, '../../.env'),          // server/.env (local override)
];
for (const envPath of candidatePaths) {
  const result = config({ path: envPath });
  if (!result.error) {
    console.log(`[GraphZep] Loaded .env from: ${envPath}`);
  }
}

const settingsSchema = z.object({
  OPENAI_API_KEY: z.string().optional(),
  SILICONFLOW_API_KEY: z.string().optional(),
  SiliconFlow_API_KEY: z.string().optional(),   // camelCase variant from main project .env
  OPENAI_BASE_URL: z.string().optional(),
  MODEL_NAME: z.string().default('deepseek-ai/DeepSeek-V3.2'),
  EMBEDDING_MODEL_NAME: z.string().default('BAAI/bge-m3'),
  NEO4J_URI: z.string().default('bolt://localhost:7687'),
  NEO4J_USER: z.string().default('neo4j'),
  NEO4J_PASSWORD: z.string().default('12345678'),
  PORT: z.string().default('3100').transform(Number),
});

/** Resolve the API key: prefer OPENAI_API_KEY, fall back to SiliconFlow variants */
function resolveApiKey(settings: z.infer<typeof settingsSchema>): string {
  const key = settings.OPENAI_API_KEY || settings.SILICONFLOW_API_KEY || settings.SiliconFlow_API_KEY;
  if (!key) {
    throw new Error('Either OPENAI_API_KEY or SiliconFlow_API_KEY must be set in .env');
  }
  return key;
}

export type Settings = z.infer<typeof settingsSchema>;

let cachedSettings: Settings | null = null;

export function getSettings(): Settings {
  if (cachedSettings) {
    return cachedSettings;
  }

  const result = settingsSchema.safeParse(process.env);
  
  if (!result.success) {
    throw new Error(`Invalid environment configuration: ${result.error.message}`);
  }

  cachedSettings = result.data;
  return cachedSettings;
}

export { resolveApiKey };
