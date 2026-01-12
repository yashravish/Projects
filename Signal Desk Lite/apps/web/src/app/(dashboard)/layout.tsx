'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { useQuery } from '@tanstack/react-query';
import { apiRequest } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Sparkles, Library, TrendingUp, LogOut } from 'lucide-react';
import { toast } from 'sonner';

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();

  const { data, isLoading, error } = useQuery({
    queryKey: ['me'],
    queryFn: () => apiRequest<{ user: { id: string; email: string } }>('/v1/auth/me'),
  });

  useEffect(() => {
    if (error || (data && data.error)) {
      router.push('/login');
    }
  }, [data, error, router]);

  const handleLogout = async () => {
    await apiRequest('/v1/auth/logout', { method: 'POST' });
    toast.success('Logged out successfully');
    router.push('/');
  };

  if (isLoading) {
    return (
      <div className="min-h-screen page-bg flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin h-12 w-12 border-4 border-primary border-t-transparent rounded-full mx-auto"></div>
          <p className="mt-4 text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen page-bg">
      <nav className="border-b border-border/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-8">
            <Link href="/collections" className="flex items-center gap-2">
              <Sparkles className="h-6 w-6 text-primary" />
              <span className="text-2xl font-display font-bold">SignalDesk</span>
            </Link>
            <div className="flex items-center gap-2">
              <Link href="/collections">
                <Button variant="ghost" size="sm">
                  <Library className="h-4 w-4 mr-2" />
                  Collections
                </Button>
              </Link>
              <Link href="/analytics">
                <Button variant="ghost" size="sm">
                  <TrendingUp className="h-4 w-4 mr-2" />
                  Analytics
                </Button>
              </Link>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-sm text-muted-foreground">{data?.data?.user.email}</span>
            <Button variant="ghost" size="sm" onClick={handleLogout}>
              <LogOut className="h-4 w-4 mr-2" />
              Logout
            </Button>
          </div>
        </div>
      </nav>
      <main className="container mx-auto px-4 py-8">{children}</main>
    </div>
  );
}
