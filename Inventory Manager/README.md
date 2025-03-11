# Inventory Manager

A modern, full-stack inventory management system built with React, TypeScript, and Supabase. This application provides a complete solution for managing products, processing orders, and handling inventory in real-time.

![Inventory Manager Screenshot](https://images.unsplash.com/photo-1553413077-190dd305871c?auto=format&fit=crop&q=80&w=2000)

## Live Demo

Visit the live application: [Inventory Manager on Netlify](https://inventory-manager-demo.netlify.app)

[![Netlify Status](https://api.netlify.com/api/v1/badges/your-netlify-badge/deploy-status)](https://app.netlify.com/sites/inventory-manager-demo/deploys)

## Features

### 🛍️ Product Management
- View all available products with real-time stock updates
- Beautiful, responsive product grid layout
- Quick add-to-cart functionality
- Stock level tracking and validation

### 🛒 Shopping Cart
- Real-time cart management
- Remove items from cart
- Clear cart functionality
- Total price calculation
- Secure checkout process

### 📦 Order Processing
- Create and track orders
- View order history
- Real-time order status updates
- Automatic stock level adjustments

### 👤 Admin Dashboard
- Secure admin-only access
- Add new products
- Update product information
- Delete products
- Monitor inventory levels

### 🔒 Security Features
- Role-based access control (RBAC)
- Row Level Security (RLS) with Supabase
- Protected admin routes
- Secure authentication

## Technology Stack

- **Frontend**: React, TypeScript
- **Styling**: Tailwind CSS
- **State Management**: Zustand
- **Database**: Supabase
- **Authentication**: Supabase Auth
- **Testing**: Playwright
- **Icons**: Lucide React
- **Notifications**: React Hot Toast
- **Hosting**: Netlify

## Getting Started

### Prerequisites

- Node.js (v18 or higher)
- npm or yarn
- Supabase account

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/inventory-manager.git
   cd inventory-manager
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create a `.env` file in the root directory:
   ```env
   VITE_SUPABASE_URL=your_supabase_url
   VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
   ```

4. Start the development server:
   ```bash
   npm run dev
   ```

### Database Setup

The application requires the following Supabase tables:

- `products`: Store product information
- `orders`: Track customer orders
- `order_items`: Link products to orders

Required policies and triggers are included in the migration files under `supabase/migrations/`.

### Running Tests

```bash
# Run all tests
npm run test

# Run tests with UI
npm run test:ui
```

### Deployment

This project is configured for automatic deployment to Netlify:

1. Connect your GitHub repository to Netlify
2. Configure build settings:
   - Build command: `npm run build`
   - Publish directory: `dist`
3. Add environment variables in Netlify dashboard:
   - `VITE_SUPABASE_URL`
   - `VITE_SUPABASE_ANON_KEY`
4. Enable automatic deployments for continuous delivery

## Project Structure

```
├── src/
│   ├── components/      # React components
│   ├── lib/            # Utility functions and configurations
│   ├── store/          # State management
│   ├── types/          # TypeScript type definitions
│   └── main.tsx        # Application entry point
├── tests/              # Playwright tests
├── supabase/           # Database migrations and configurations
└── public/             # Static assets
```

## Security Considerations

- All database operations are protected by Row Level Security (RLS)
- Admin functions are restricted to users with admin role
- Stock levels are protected from going negative
- Transactions ensure data consistency
- Environment variables are securely managed in Netlify

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Supabase](https://supabase.io/) for the amazing backend service
- [Tailwind CSS](https://tailwindcss.com/) for the utility-first CSS framework
- [Lucide](https://lucide.dev/) for the beautiful icons
- [Netlify](https://www.netlify.com/) for hosting and continuous deployment