export interface User {
  id: number;
  email: string;
  full_name: string;
  department: string;
  role: string;
}

export interface ProcessRequest {
  id: number;
  requester_id: number;
  requester_name: string;
  title: string;
  description: string;
  category: string;
  urgency: number;
  business_impact: number;
  desired_completion_date: string | null;
  status: string;
  priority_score: number | null;
  assigned_team: string | null;
  assigned_owner: string | null;
  created_at: string;
  updated_at: string;
}

export interface RoutingDecision {
  id: number;
  suggested_team: string;
  priority_score: number;
  routing_explanation: string;
  category_match: string;
  created_at: string;
}

export interface AISummary {
  id: number;
  request_id: number;
  summary: string;
  business_impact_explanation: string;
  recommended_action: string;
  leadership_summary: string;
  implementation_notes: string | null;
  provider_used: string;
  created_at: string;
}

export interface RequestUpdate {
  id: number;
  author_name: string;
  status_change: string | null;
  note: string | null;
  created_at: string;
}

export interface RequestDetail extends ProcessRequest {
  routing_decision: RoutingDecision | null;
  ai_summary: AISummary | null;
  updates: RequestUpdate[];
}

export interface AnalyticsOverview {
  total_requests: number;
  open_requests: number;
  closed_requests: number;
  avg_priority: number;
  requests_this_week: number;
}

export interface CategoryCount {
  category: string;
  count: number;
}

export interface DepartmentCount {
  department: string;
  count: number;
}

export interface PriorityCount {
  priority_range: string;
  count: number;
}

export interface StatusCount {
  status: string;
  count: number;
}

export interface PainPoint {
  description: string;
  count: number;
  category: string;
}
