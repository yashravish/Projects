import { useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { createRequest } from '../api/client';

const categories = [
  { value: 'access_request', label: 'Access Request' },
  { value: 'workflow_issue', label: 'Workflow Issue' },
  { value: 'data_correction', label: 'Data Correction' },
  { value: 'report_request', label: 'Report Request' },
  { value: 'automation_idea', label: 'Automation Idea' },
  { value: 'process_bottleneck', label: 'Process Bottleneck' },
];

const urgencyLevels = [
  { value: 1, label: '1 — Low' },
  { value: 2, label: '2 — Moderate' },
  { value: 3, label: '3 — Medium' },
  { value: 4, label: '4 — High' },
  { value: 5, label: '5 — Critical' },
];

const impactLevels = [
  { value: 1, label: '1 — Minimal' },
  { value: 2, label: '2 — Low' },
  { value: 3, label: '3 — Moderate' },
  { value: 4, label: '4 — Significant' },
  { value: 5, label: '5 — Critical' },
];

const inputClass =
  'block w-full rounded-md border border-gray-300 px-3 py-2.5 text-sm shadow-sm placeholder-gray-400 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500';

export default function NewRequest() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [category, setCategory] = useState('');
  const [urgency, setUrgency] = useState('');
  const [businessImpact, setBusinessImpact] = useState('');
  const [desiredDate, setDesiredDate] = useState('');

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const result = await createRequest({
        title,
        description,
        category,
        urgency: Number(urgency),
        business_impact: Number(businessImpact),
        desired_completion_date: desiredDate || null,
      });
      navigate(`/requests/${result.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit request');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Submit New Request</h1>
        <p className="mt-1 text-sm text-gray-500">
          Describe the process issue or improvement you need
        </p>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 sm:p-8">
        {error && (
          <div className="mb-6 rounded-md bg-red-50 border border-red-200 p-3">
            <p className="text-sm text-red-700">{error}</p>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label htmlFor="title" className="block text-sm font-medium text-gray-700 mb-1">
              Title <span className="text-red-500">*</span>
            </label>
            <input
              id="title"
              type="text"
              required
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className={inputClass}
              placeholder="Brief summary of the request"
            />
          </div>

          <div>
            <label htmlFor="description" className="block text-sm font-medium text-gray-700 mb-1">
              Description <span className="text-red-500">*</span>
            </label>
            <textarea
              id="description"
              required
              rows={4}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className={inputClass}
              placeholder="Provide details about the issue, current process, and desired outcome..."
            />
          </div>

          <div>
            <label htmlFor="category" className="block text-sm font-medium text-gray-700 mb-1">
              Category <span className="text-red-500">*</span>
            </label>
            <select
              id="category"
              required
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className={inputClass}
            >
              <option value="">Select a category</option>
              {categories.map((c) => (
                <option key={c.value} value={c.value}>{c.label}</option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            <div>
              <label htmlFor="urgency" className="block text-sm font-medium text-gray-700 mb-1">
                Urgency <span className="text-red-500">*</span>
              </label>
              <select
                id="urgency"
                required
                value={urgency}
                onChange={(e) => setUrgency(e.target.value)}
                className={inputClass}
              >
                <option value="">Select urgency</option>
                {urgencyLevels.map((u) => (
                  <option key={u.value} value={u.value}>{u.label}</option>
                ))}
              </select>
            </div>

            <div>
              <label htmlFor="impact" className="block text-sm font-medium text-gray-700 mb-1">
                Business Impact <span className="text-red-500">*</span>
              </label>
              <select
                id="impact"
                required
                value={businessImpact}
                onChange={(e) => setBusinessImpact(e.target.value)}
                className={inputClass}
              >
                <option value="">Select impact</option>
                {impactLevels.map((i) => (
                  <option key={i.value} value={i.value}>{i.label}</option>
                ))}
              </select>
            </div>
          </div>

          <div>
            <label htmlFor="date" className="block text-sm font-medium text-gray-700 mb-1">
              Desired Completion Date <span className="text-gray-400 text-xs">(optional)</span>
            </label>
            <input
              id="date"
              type="date"
              value={desiredDate}
              onChange={(e) => setDesiredDate(e.target.value)}
              className={inputClass}
            />
          </div>

          <div className="pt-2">
            <button
              type="submit"
              disabled={loading}
              className="w-full flex items-center justify-center rounded-md bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? (
                <>
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent mr-2" />
                  Submitting...
                </>
              ) : (
                'Submit Request'
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
