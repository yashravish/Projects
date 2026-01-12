import type { Metadata } from 'next';
import { Playfair_Display, IBM_Plex_Sans } from 'next/font/google';
import './globals.css';
import { Providers } from '@/components/providers';
import { Toaster } from 'sonner';

const playfair = Playfair_Display({
  subsets: ['latin'],
  variable: '--font-display',
  display: 'swap',
});

const ibmPlexSans = IBM_Plex_Sans({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  variable: '--font-body',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'SignalDesk Lite - AI Document Intelligence',
  description: 'Transform your documents into intelligent, queryable knowledge bases',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${playfair.variable} ${ibmPlexSans.variable} antialiased`}>
        <Providers>{children}</Providers>
        <Toaster position="top-right" />
      </body>
    </html>
  );
}
