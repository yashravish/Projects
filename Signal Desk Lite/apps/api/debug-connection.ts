import { PrismaClient } from '@prisma/client';
import dotenv from 'dotenv';
import path from 'path';

// Explicitly load .env from current directory
dotenv.config({ path: path.resolve(process.cwd(), '.env') });

console.log('--- DIAGNOSTIC START ---');
console.log('Checking environment variables...');

const url = process.env.DATABASE_URL;
if (!url) {
    console.error('ERROR: DATABASE_URL is not set in environment variables.');
} else {
    // Mask password in output
    const masked = url.replace(/:([^:@]+)@/, ':****@');
    console.log(`DATABASE_URL is set: ${masked}`);
}

console.log('Attempting Prisma connection...');
const prisma = new PrismaClient();

async function main() {
    try {
        await prisma.$connect();
        console.log('SUCCESS: Connected to database.');

        // Try a simple query
        const count = await prisma.user.count();
        console.log(`User count: ${count}`);

    } catch (e: any) {
        console.error('ERROR: Could not connect to database.');
        console.error(e.message);
    } finally {
        await prisma.$disconnect();
    }
}
main();
