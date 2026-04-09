interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  color?: string;
}

const colorMap: Record<string, string> = {
  indigo: 'border-indigo-500',
  blue: 'border-blue-500',
  green: 'border-green-500',
  yellow: 'border-yellow-500',
  red: 'border-red-500',
  purple: 'border-purple-500',
  teal: 'border-teal-500',
};

export default function StatCard({ title, value, subtitle, color = 'indigo' }: StatCardProps) {
  const borderClass = colorMap[color] || 'border-indigo-500';

  return (
    <div className={`bg-white rounded-lg shadow-sm border-l-4 ${borderClass} p-6`}>
      <p className="text-sm font-medium text-gray-500 truncate">{title}</p>
      <p className="mt-2 text-3xl font-bold text-gray-900">{value}</p>
      {subtitle && <p className="mt-1 text-sm text-gray-500">{subtitle}</p>}
    </div>
  );
}
