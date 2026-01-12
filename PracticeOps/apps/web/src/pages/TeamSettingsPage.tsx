/**
 * Team Settings Page
 *
 * Route: /settings
 * Access: ADMIN only
 *
 * Allows team admins to:
 * - Edit team name
 * - Manage team members (edit role/section, remove)
 * - Manage invites (create, list, revoke)
 */

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { TableSkeleton } from "@/components/ui/loading";
import { NoTeamMembersEmptyState, NoInvitesEmptyState } from "@/components/ui/empty-state";
import { toastSuccess, toastError } from "@/lib/toast";
import {
  Users,
  Mail,
  Pencil,
  Trash2,
  UserPlus,
  Check,
  AlertCircle,
  Building,
  Calendar,
  Plus,
} from "lucide-react";
import { useAuth } from "@/lib/auth";
import {
  updateTeam,
  getTeamMembers,
  updateMember,
  removeMember,
  createInvite,
  listInvites,
  revokeInvite,
  createCycle,
  listCycles,
  type CreateCycleRequest,
} from "@/lib/api/client";
import type {
  MemberResponse,
  InviteResponse,
  CycleResponse,
  Role,
} from "@/lib/api/types";
import { format, parseISO, addDays } from "date-fns";

const ROLES: Role[] = ["MEMBER", "SECTION_LEADER", "ADMIN"];
const SECTIONS = ["Soprano", "Alto", "Tenor", "Bass"];

export function TeamSettingsPage() {
  const { primaryTeam, user } = useAuth();
  const queryClient = useQueryClient();

  const [teamName, setTeamName] = useState("");
  const [isEditingName, setIsEditingName] = useState(false);
  const [editingMember, setEditingMember] = useState<MemberResponse | null>(null);
  const [memberToRemove, setMemberToRemove] = useState<MemberResponse | null>(null);
  const [inviteToRevoke, setInviteToRevoke] = useState<InviteResponse | null>(null);
  const [showCreateInvite, setShowCreateInvite] = useState(false);
  const [copiedInvite, setCopiedInvite] = useState<string | null>(null);

  // Form state for editing member
  const [editRole, setEditRole] = useState<Role>("MEMBER");
  const [editSection, setEditSection] = useState<string>("");

  // Form state for creating invite
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState<Role>("MEMBER");
  const [inviteSection, setInviteSection] = useState<string>("");

  // Cycle state
  const [showCreateCycle, setShowCreateCycle] = useState(false);
  const [cycleDate, setCycleDate] = useState(() => {
    // Default to next Sunday
    const today = new Date();
    const daysUntilSunday = (7 - today.getDay()) % 7 || 7;
    return format(addDays(today, daysUntilSunday), "yyyy-MM-dd");
  });
  const [cycleName, setCycleName] = useState("");

  // Fetch team members
  const {
    data: membersData,
    isLoading: membersLoading,
  } = useQuery({
    queryKey: ["team-members", primaryTeam?.team_id],
    queryFn: () => getTeamMembers(primaryTeam!.team_id),
    enabled: !!primaryTeam?.team_id,
  });

  // Fetch invites
  const {
    data: invitesData,
    isLoading: invitesLoading,
  } = useQuery({
    queryKey: ["team-invites", primaryTeam?.team_id],
    queryFn: () => listInvites(primaryTeam!.team_id),
    enabled: !!primaryTeam?.team_id,
  });

  // Fetch cycles
  const {
    data: cyclesData,
    isLoading: cyclesLoading,
  } = useQuery({
    queryKey: ["team-cycles", primaryTeam?.team_id],
    queryFn: () => listCycles(primaryTeam!.team_id),
    enabled: !!primaryTeam?.team_id,
  });

  // Update team mutation
  const updateTeamMutation = useMutation({
    mutationFn: (name: string) => updateTeam(primaryTeam!.team_id, { name }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["me"] });
      setIsEditingName(false);
      toastSuccess("Team name updated");
    },
    onError: (err) => {
      toastError(err instanceof Error ? err.message : "Failed to update team");
    },
  });

  // Update member mutation
  const updateMemberMutation = useMutation({
    mutationFn: ({ userId, role, section }: { userId: string; role?: Role; section?: string }) =>
      updateMember(primaryTeam!.team_id, userId, { role, section }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["team-members"] });
      setEditingMember(null);
      toastSuccess("Member updated");
    },
    onError: (err) => {
      toastError(err instanceof Error ? err.message : "Failed to update member");
    },
  });

  // Remove member mutation
  const removeMemberMutation = useMutation({
    mutationFn: (userId: string) => removeMember(primaryTeam!.team_id, userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["team-members"] });
      setMemberToRemove(null);
      toastSuccess("Member removed");
    },
    onError: (err) => {
      toastError(err instanceof Error ? err.message : "Failed to remove member");
    },
  });

  // Create invite mutation
  const createInviteMutation = useMutation({
    mutationFn: () =>
      createInvite(primaryTeam!.team_id, {
        email: inviteEmail || undefined,
        role: inviteRole,
        section: inviteSection || undefined,
      }),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["team-invites"] });
      setShowCreateInvite(false);
      setInviteEmail("");
      setInviteRole("MEMBER");
      setInviteSection("");
      // Copy invite link to clipboard
      navigator.clipboard.writeText(data.invite_link);
      setCopiedInvite(data.invite_link);
      setTimeout(() => setCopiedInvite(null), 5000);
      toastSuccess("Invite created and link copied to clipboard");
    },
    onError: (err) => {
      toastError(err instanceof Error ? err.message : "Failed to create invite");
    },
  });

  // Revoke invite mutation
  const revokeInviteMutation = useMutation({
    mutationFn: (inviteId: string) => revokeInvite(inviteId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["team-invites"] });
      setInviteToRevoke(null);
      toastSuccess("Invite revoked");
    },
    onError: (err) => {
      toastError(err instanceof Error ? err.message : "Failed to revoke invite");
    },
  });

  // Create cycle mutation
  const createCycleMutation = useMutation({
    mutationFn: (data: CreateCycleRequest) => createCycle(primaryTeam!.team_id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["team-cycles"] });
      queryClient.invalidateQueries({ queryKey: ["active-cycle"] });
      setShowCreateCycle(false);
      setCycleName("");
      // Reset date to next Sunday
      const today = new Date();
      const daysUntilSunday = (7 - today.getDay()) % 7 || 7;
      setCycleDate(format(addDays(today, daysUntilSunday), "yyyy-MM-dd"));
      toastSuccess("Rehearsal cycle created!");
    },
    onError: (err) => {
      toastError(err instanceof Error ? err.message : "Failed to create cycle");
    },
  });

  const handleEditMember = (member: MemberResponse) => {
    setEditingMember(member);
    setEditRole(member.role);
    setEditSection(member.section || "");
  };

  // Check if current user is admin
  if (primaryTeam?.role !== "ADMIN") {
    return (
      <div className="container mx-auto px-4 md:px-6 lg:px-8 pt-6 pb-6 space-y-6 max-w-7xl">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Team Settings</h1>
        </div>
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            You don't have permission to access team settings. Only team admins can manage the team.
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 md:px-6 lg:px-8 pt-6 pb-6 space-y-6 max-w-7xl">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Team Settings</h1>
        <p className="text-muted-foreground">
          Manage your team, members, and invitations
        </p>
      </div>

      {/* Copied notification */}
      {copiedInvite && (
        <Alert className="border-emerald-500 bg-emerald-50 dark:bg-emerald-950/20">
          <Check className="h-4 w-4 text-emerald-500" />
          <AlertDescription className="text-emerald-700 dark:text-emerald-300">
            Invite link copied to clipboard! Share it with the new member.
          </AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="team" className="space-y-4">
        <TabsList>
          <TabsTrigger value="team" className="gap-2">
            <Building className="h-4 w-4" />
            Team Info
          </TabsTrigger>
          <TabsTrigger value="cycles" className="gap-2">
            <Calendar className="h-4 w-4" />
            Rehearsals
          </TabsTrigger>
          <TabsTrigger value="members" className="gap-2">
            <Users className="h-4 w-4" />
            Members
          </TabsTrigger>
          <TabsTrigger value="invites" className="gap-2">
            <Mail className="h-4 w-4" />
            Invites
          </TabsTrigger>
        </TabsList>

        {/* Team Info Tab */}
        <TabsContent value="team" className="space-y-4">
          <Card>
            <CardHeader className="py-4">
              <CardTitle>Team Information</CardTitle>
              <CardDescription>Basic information about your team</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 pb-4">
              <div className="space-y-2">
                <Label>Team Name</Label>
                {isEditingName ? (
                  <div className="flex gap-2">
                    <Input
                      value={teamName}
                      onChange={(e) => setTeamName(e.target.value)}
                      placeholder="Enter team name"
                    />
                    <Button
                      onClick={() => updateTeamMutation.mutate(teamName)}
                      disabled={updateTeamMutation.isPending || !teamName.trim()}
                    >
                      Save
                    </Button>
                    <Button variant="outline" onClick={() => setIsEditingName(false)}>
                      Cancel
                    </Button>
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <p className="text-lg font-medium">Team</p>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => {
                        setTeamName("");
                        setIsEditingName(true);
                      }}
                    >
                      <Pencil className="h-4 w-4" />
                    </Button>
                  </div>
                )}
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-1">
                  <Label className="text-muted-foreground">Members</Label>
                  <p className="text-2xl font-bold">{membersData?.items.length || 0}</p>
                </div>
                <div className="space-y-1">
                  <Label className="text-muted-foreground">Active Invites</Label>
                  <p className="text-2xl font-bold">
                    {invitesData?.items.filter((i) => !i.used_at).length || 0}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Cycles Tab */}
        <TabsContent value="cycles" className="space-y-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between py-4">
              <div>
                <CardTitle>Rehearsal Cycles</CardTitle>
                <CardDescription>
                  Schedule rehearsals to enable practice tracking
                </CardDescription>
              </div>
              <Button onClick={() => setShowCreateCycle(true)}>
                <Plus className="h-4 w-4 mr-2" />
                New Rehearsal
              </Button>
            </CardHeader>
            <CardContent className="pb-4">
              {cyclesLoading ? (
                <TableSkeleton rows={3} />
              ) : cyclesData?.items.length === 0 ? (
                <div className="text-center py-8">
                  <Calendar className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                  <h3 className="font-semibold mb-2">No rehearsals scheduled</h3>
                  <p className="text-muted-foreground mb-4">
                    Create your first rehearsal cycle to enable practice logging
                  </p>
                  <Button onClick={() => setShowCreateCycle(true)}>
                    <Plus className="h-4 w-4 mr-2" />
                    Schedule Rehearsal
                  </Button>
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Date</TableHead>
                      <TableHead>Name</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Created</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {cyclesData?.items.map((cycle: CycleResponse) => {
                      const cycleDate = parseISO(cycle.date);
                      const isUpcoming = cycleDate >= new Date();
                      const isPast = cycleDate < new Date();
                      return (
                        <TableRow key={cycle.id}>
                          <TableCell className="font-medium">
                            {format(cycleDate, "EEEE, MMM d, yyyy")}
                          </TableCell>
                          <TableCell>{cycle.name || "—"}</TableCell>
                          <TableCell>
                            {isUpcoming ? (
                              <Badge className="bg-emerald-500/15 text-emerald-600 border-emerald-500/30">
                                Upcoming
                              </Badge>
                            ) : isPast ? (
                              <Badge variant="secondary">Past</Badge>
                            ) : (
                              <Badge>Active</Badge>
                            )}
                          </TableCell>
                          <TableCell className="text-muted-foreground">
                            {format(parseISO(cycle.created_at), "MMM d, yyyy")}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Members Tab */}
        <TabsContent value="members" className="space-y-4">
          <Card>
            <CardHeader className="py-4">
              <CardTitle>Team Members</CardTitle>
              <CardDescription>Manage member roles and sections</CardDescription>
            </CardHeader>
            <CardContent className="pb-4">
              {membersLoading ? (
                <TableSkeleton rows={3} />
              ) : membersData?.items.length === 0 ? (
                <NoTeamMembersEmptyState showAction onAction={() => setShowCreateInvite(true)} />
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Email</TableHead>
                      <TableHead>Role</TableHead>
                      <TableHead>Section</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {membersData?.items.map((member) => (
                      <TableRow key={member.id}>
                        <TableCell className="font-medium">
                          {member.display_name}
                          {member.user_id === user?.id && (
                            <Badge variant="outline" className="ml-2">You</Badge>
                          )}
                        </TableCell>
                        <TableCell>{member.email}</TableCell>
                        <TableCell>
                          <Badge
                            variant={
                              member.role === "ADMIN"
                                ? "default"
                                : member.role === "SECTION_LEADER"
                                  ? "secondary"
                                  : "outline"
                            }
                          >
                            {member.role}
                          </Badge>
                        </TableCell>
                        <TableCell>{member.section || "—"}</TableCell>
                        <TableCell className="text-right">
                          <div className="flex justify-end gap-2">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleEditMember(member)}
                            >
                              <Pencil className="h-4 w-4" />
                            </Button>
                            {member.user_id !== user?.id && (
                              <Button
                                variant="ghost"
                                size="sm"
                                className="text-destructive hover:text-destructive"
                                onClick={() => setMemberToRemove(member)}
                              >
                                <Trash2 className="h-4 w-4" />
                              </Button>
                            )}
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Invites Tab */}
        <TabsContent value="invites" className="space-y-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between py-4">
              <div>
                <CardTitle>Team Invites</CardTitle>
                <CardDescription>Manage pending invitations</CardDescription>
              </div>
              <Button onClick={() => setShowCreateInvite(true)}>
                <UserPlus className="h-4 w-4 mr-2" />
                Create Invite
              </Button>
            </CardHeader>
            <CardContent className="pb-4">
              {invitesLoading ? (
                <TableSkeleton rows={2} />
              ) : invitesData?.items.length === 0 ? (
                <NoInvitesEmptyState showAction onAction={() => setShowCreateInvite(true)} />
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Email</TableHead>
                      <TableHead>Role</TableHead>
                      <TableHead>Section</TableHead>
                      <TableHead>Expires</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {invitesData?.items.map((invite) => {
                      const isExpired = new Date(invite.expires_at) < new Date();
                      return (
                        <TableRow key={invite.id} className={isExpired ? "opacity-50" : ""}>
                          <TableCell>{invite.email || "Anyone"}</TableCell>
                          <TableCell>
                            <Badge variant="outline">{invite.role}</Badge>
                          </TableCell>
                          <TableCell>{invite.section || "—"}</TableCell>
                          <TableCell>
                            {isExpired ? (
                              <Badge variant="destructive">Expired</Badge>
                            ) : (
                              new Date(invite.expires_at).toLocaleDateString()
                            )}
                          </TableCell>
                          <TableCell className="text-right">
                            {!isExpired && (
                              <Button
                                variant="ghost"
                                size="sm"
                                className="text-destructive hover:text-destructive"
                                onClick={() => setInviteToRevoke(invite)}
                              >
                                <Trash2 className="h-4 w-4" />
                              </Button>
                            )}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Edit Member Dialog */}
      <Dialog open={!!editingMember} onOpenChange={(open) => !open && setEditingMember(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Member</DialogTitle>
            <DialogDescription>
              Update role and section for {editingMember?.display_name}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Role</Label>
              <Select value={editRole} onValueChange={(v: string) => setEditRole(v as Role)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {ROLES.map((role) => (
                    <SelectItem key={role} value={role}>
                      {role}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Section</Label>
              <Select value={editSection} onValueChange={setEditSection}>
                <SelectTrigger>
                  <SelectValue placeholder="Select section" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">No section</SelectItem>
                  {SECTIONS.map((section) => (
                    <SelectItem key={section} value={section}>
                      {section}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setEditingMember(null)}>
              Cancel
            </Button>
            <Button
              onClick={() => {
                if (editingMember) {
                  updateMemberMutation.mutate({
                    userId: editingMember.user_id,
                    role: editRole,
                    section: editSection || undefined,
                  });
                }
              }}
              disabled={updateMemberMutation.isPending}
            >
              {updateMemberMutation.isPending ? "Saving..." : "Save Changes"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Remove Member Confirmation */}
      <Dialog open={!!memberToRemove} onOpenChange={(open) => !open && setMemberToRemove(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Remove Member</DialogTitle>
            <DialogDescription>
              Are you sure you want to remove {memberToRemove?.display_name} from the team?
              This action cannot be undone.
            </DialogDescription>
          </DialogHeader>

          <DialogFooter>
            <Button variant="outline" onClick={() => setMemberToRemove(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                if (memberToRemove) {
                  removeMemberMutation.mutate(memberToRemove.user_id);
                }
              }}
              disabled={removeMemberMutation.isPending}
            >
              {removeMemberMutation.isPending ? "Removing..." : "Remove Member"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Create Invite Dialog */}
      <Dialog open={showCreateInvite} onOpenChange={setShowCreateInvite}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create Invite</DialogTitle>
            <DialogDescription>
              Generate an invite link to add a new team member
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Email (optional)</Label>
              <Input
                type="email"
                value={inviteEmail}
                onChange={(e) => setInviteEmail(e.target.value)}
                placeholder="newmember@example.com"
              />
              <p className="text-xs text-muted-foreground">
                Leave empty to create a generic invite link
              </p>
            </div>

            <div className="space-y-2">
              <Label>Role</Label>
              <Select value={inviteRole} onValueChange={(v: string) => setInviteRole(v as Role)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {ROLES.map((role) => (
                    <SelectItem key={role} value={role}>
                      {role}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Section (optional)</Label>
              <Select value={inviteSection} onValueChange={setInviteSection}>
                <SelectTrigger>
                  <SelectValue placeholder="Select section" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">No section</SelectItem>
                  {SECTIONS.map((section) => (
                    <SelectItem key={section} value={section}>
                      {section}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setShowCreateInvite(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => createInviteMutation.mutate()}
              disabled={createInviteMutation.isPending}
            >
              {createInviteMutation.isPending ? "Creating..." : "Create Invite"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Revoke Invite Confirmation */}
      <Dialog open={!!inviteToRevoke} onOpenChange={(open) => !open && setInviteToRevoke(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Revoke Invite</DialogTitle>
            <DialogDescription>
              Are you sure you want to revoke this invite? The link will no longer work.
            </DialogDescription>
          </DialogHeader>

          <DialogFooter>
            <Button variant="outline" onClick={() => setInviteToRevoke(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                if (inviteToRevoke) {
                  revokeInviteMutation.mutate(inviteToRevoke.id);
                }
              }}
              disabled={revokeInviteMutation.isPending}
            >
              {revokeInviteMutation.isPending ? "Revoking..." : "Revoke Invite"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Create Cycle Dialog */}
      <Dialog open={showCreateCycle} onOpenChange={setShowCreateCycle}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Schedule Rehearsal</DialogTitle>
            <DialogDescription>
              Create a new rehearsal cycle. Team members can log practice and track issues leading up to this date.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="cycle-date">Rehearsal Date</Label>
              <Input
                id="cycle-date"
                type="date"
                value={cycleDate}
                onChange={(e) => setCycleDate(e.target.value)}
                min={format(new Date(), "yyyy-MM-dd")}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="cycle-name">Name (optional)</Label>
              <Input
                id="cycle-name"
                value={cycleName}
                onChange={(e) => setCycleName(e.target.value)}
                placeholder="e.g., Week 1, Spring Concert Prep"
              />
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setShowCreateCycle(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => {
                createCycleMutation.mutate({
                  date: cycleDate,
                  name: cycleName || undefined,
                });
              }}
              disabled={createCycleMutation.isPending || !cycleDate}
            >
              {createCycleMutation.isPending ? "Creating..." : "Create Rehearsal"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

