# Responsive Behavior Patterns

This document defines the responsive design patterns used across PracticeOps.
All patterns use Tailwind CSS default breakpoints.

## Breakpoints

| Breakpoint | Min Width | Typical Use |
|------------|-----------|-------------|
| `sm` | 640px | Small tablets, large phones |
| `md` | 768px | Tablets, small laptops |
| `lg` | 1024px | Laptops, small desktops |
| `xl` | 1280px+ | Large desktops |

## Component Patterns

### 1. AppShell Navigation

**Behavior:**
- `< md`: Bottom nav or hamburger menu (Sheet-based)
- `>= md`: Fixed sidebar navigation

**Implementation:**

```tsx
// Sidebar visibility
<Sidebar className="hidden md:flex" />

// Mobile hamburger trigger
<Button className="md:hidden">
  <Menu className="h-5 w-5" />
</Button>

// Mobile nav as Sheet
<Sheet>
  <SheetTrigger asChild>
    <Button variant="ghost" size="icon" className="md:hidden">
      <Menu className="h-5 w-5" />
    </Button>
  </SheetTrigger>
  <SheetContent side="left">
    {/* Navigation content */}
  </SheetContent>
</Sheet>
```

### 2. Dashboard Stats Row

**Behavior:**
- `< sm`: Vertical stack (single column)
- `sm–md`: 2×2 grid
- `>= md`: Horizontal row (3 columns)

**Implementation:**

```tsx
// Stats container with responsive grid
<div className="grid gap-4 grid-cols-1 sm:grid-cols-2 md:grid-cols-3">
  <StatsCard />
  <StatsCard />
  <StatsCard />
</div>

// Alternative: horizontal scroll on mobile
<div className="flex gap-4 overflow-x-auto pb-2 -mx-4 px-4 md:mx-0 md:px-0 md:grid md:grid-cols-3">
  <div className="min-w-[200px] md:min-w-0">
    <StatsCard />
  </div>
  {/* ... more cards */}
</div>
```

### 3. Tables vs Card Lists

**Behavior:**
- `< md`: Card list (stacked cards)
- `>= md`: Traditional table layout

**Implementation:**

```tsx
// Table visible on md+
<div className="hidden md:block">
  <Table>
    {/* Table content */}
  </Table>
</div>

// Card list visible on mobile
<div className="md:hidden space-y-4">
  {items.map(item => (
    <Card key={item.id}>
      {/* Card content with stacked layout */}
    </Card>
  ))}
</div>
```

### 4. Filter Bars

**Behavior:**
- `< md`: Filters button that opens a Sheet
- `>= md`: Inline filter controls

**Implementation:**

```tsx
// Desktop: inline filters
<div className="hidden md:flex gap-4">
  <Select>{/* Filter options */}</Select>
  <Select>{/* More filters */}</Select>
</div>

// Mobile: Sheet-based filters
<div className="md:hidden">
  <Sheet>
    <SheetTrigger asChild>
      <Button variant="outline" className="w-full">
        <Filter className="h-4 w-4 mr-2" />
        Filters
        {hasActiveFilters && (
          <Badge variant="secondary" className="ml-2">
            {filterCount}
          </Badge>
        )}
      </Button>
    </SheetTrigger>
    <SheetContent>
      <SheetHeader>
        <SheetTitle>Filter</SheetTitle>
      </SheetHeader>
      <div className="mt-4 space-y-4">
        {/* Filter controls */}
      </div>
    </SheetContent>
  </Sheet>
</div>
```

### 5. Dialogs vs Bottom Sheets

**Behavior:**
- `< md`: Bottom Sheet (slides from bottom)
- `>= md`: Centered Dialog

**Implementation:**

```tsx
// Using isMobile hook
function useIsMobile() {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile();
    window.addEventListener("resize", checkMobile);
    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  return isMobile;
}

// Component usage
const isMobile = useIsMobile();

if (isMobile) {
  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetContent side="bottom" className="h-[85vh]">
        {/* Form content */}
      </SheetContent>
    </Sheet>
  );
}

return (
  <Dialog open={open} onOpenChange={setOpen}>
    <DialogContent className="sm:max-w-[425px]">
      {/* Form content */}
    </DialogContent>
  </Dialog>
);
```

### 6. Two-Column Layouts

**Behavior:**
- `< lg`: Single column (stacked)
- `>= lg`: Two columns side-by-side

**Implementation:**

```tsx
// Main content + sidebar layout
<div className="grid gap-6 lg:grid-cols-3">
  {/* Main content - 2/3 width on desktop */}
  <div className="lg:col-span-2">
    <MainContent />
  </div>
  
  {/* Sidebar - 1/3 width on desktop */}
  <div>
    <Sidebar />
  </div>
</div>

// Equal columns layout
<div className="grid gap-6 lg:grid-cols-2">
  <div>Column 1</div>
  <div>Column 2</div>
</div>
```

## Utility Patterns

### Responsive Padding

```tsx
// Page padding - tighter on mobile
<div className="p-4 md:p-6">

// Container with responsive max-width
<div className="container mx-auto px-4 md:px-6">
```

### Responsive Typography

```tsx
// Larger headings on desktop
<h1 className="text-2xl md:text-3xl font-bold">

// Text that wraps differently
<p className="text-sm md:text-base">
```

### Responsive Spacing

```tsx
// Gap adjustments
<div className="gap-4 md:gap-6">

// Margin adjustments
<div className="mt-4 md:mt-6">
```

### Hiding/Showing Elements

```tsx
// Hide on mobile, show on tablet+
<div className="hidden md:block">

// Show on mobile, hide on tablet+
<div className="md:hidden">

// Hide on mobile, flex on tablet+
<div className="hidden md:flex">
```

## Component-Specific Notes

### Toasts

Toasts automatically adjust position based on viewport:
- Desktop: Bottom-right (via ToastViewport)
- Mobile: Bottom-center (via ToastViewport responsive classes)

### Loading States

Use appropriate loading patterns for viewport:
- Full-page: `PageLoader` centered vertically
- Lists: `ListSkeleton` with responsive grid matching content
- Buttons: `ButtonSpinner` inline

### Empty States

Empty states should maintain consistent padding but can adjust icon size:
- Use `compact` prop for inline empty states
- Full empty states use `Card` container

## Testing Responsive Behavior

Use these viewport sizes for testing:

| Device | Width | Height |
|--------|-------|--------|
| Mobile S | 320px | 568px |
| Mobile M | 375px | 667px |
| Mobile L | 425px | 812px |
| Tablet | 768px | 1024px |
| Laptop | 1024px | 768px |
| Desktop | 1440px | 900px |

Browser DevTools responsive mode or Playwright viewport tests can verify behavior.

