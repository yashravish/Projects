'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { FileText, Sparkles, Search, TrendingUp } from 'lucide-react';

export default function HomePage() {
  return (
    <div className="min-h-screen page-bg">
      <nav className="border-b border-border/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center gap-2"
          >
            <Sparkles className="h-6 w-6 text-primary" />
            <span className="text-2xl font-display font-bold">SignalDesk</span>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center gap-4"
          >
            <Link href="/login">
              <Button variant="ghost">Login</Button>
            </Link>
            <Link href="/signup">
              <Button>Get Started</Button>
            </Link>
          </motion.div>
        </div>
      </nav>

      <main className="container mx-auto px-4 py-20">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="max-w-4xl mx-auto text-center space-y-6"
        >
          <h1 className="text-6xl md:text-7xl font-display font-bold leading-tight">
            Transform Documents
            <br />
            Into{' '}
            <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              Intelligence
            </span>
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Upload your documents, query them with natural language, and get AI-powered insights
            with precise citations and executive summaries.
          </p>
          <div className="flex items-center justify-center gap-4 pt-4">
            <Link href="/signup">
              <Button size="lg" className="text-lg">
                Start Free <Sparkles className="ml-2 h-5 w-5" />
              </Button>
            </Link>
            <Link href="/login">
              <Button size="lg" variant="outline" className="text-lg">
                Sign In
              </Button>
            </Link>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="grid md:grid-cols-3 gap-6 mt-20 max-w-5xl mx-auto"
        >
          <FeatureCard
            icon={<FileText className="h-8 w-8" />}
            title="Upload & Index"
            description="Drop in PDFs, markdown, or text files. We'll chunk and embed them automatically."
          />
          <FeatureCard
            icon={<Search className="h-8 w-8" />}
            title="Query with AI"
            description="Ask questions in natural language. Get answers with inline citations and sources."
          />
          <FeatureCard
            icon={<TrendingUp className="h-8 w-8" />}
            title="Track Analytics"
            description="Monitor usage, token consumption, and estimated costs across all collections."
          />
        </motion.div>
      </main>
    </div>
  );
}

function FeatureCard({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      className="glass rounded-xl p-8 card-hover"
    >
      <div className="text-primary mb-4">{icon}</div>
      <h3 className="text-xl font-display font-semibold mb-2">{title}</h3>
      <p className="text-muted-foreground">{description}</p>
    </motion.div>
  );
}
