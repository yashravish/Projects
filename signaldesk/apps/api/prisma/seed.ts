import { PrismaClient } from '@prisma/client';
import bcrypt from 'bcrypt';
import { CONFIG } from '@signaldesk/shared';

const prisma = new PrismaClient();

async function main() {
  console.log('Seeding database...');

  const passwordHash = await bcrypt.hash('password123', CONFIG.BCRYPT_ROUNDS);

  const user = await prisma.user.upsert({
    where: { email: 'demo@signaldesk.com' },
    update: {},
    create: {
      email: 'demo@signaldesk.com',
      passwordHash,
    },
  });

  console.log(`Created user: ${user.email}`);

  const collection = await prisma.collection.create({
    data: {
      userId: user.id,
      name: 'Demo Collection',
      description: 'A sample collection for testing',
    },
  });

  console.log(`Created collection: ${collection.name}`);

  console.log('Seed completed!');
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
