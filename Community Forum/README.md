# A Community Forum

A modern, real-time community forum built with React, Supabase, and TypeScript. Features a beautiful glass-morphism design, infinite scrolling, and real-time interactions.

## Live Demo

Visit the live demo: [Community Forum](https://endearing-stroopwafel-3da573.netlify.app)

[![Netlify Status](https://api.netlify.com/api/v1/badges/3da573/deploy-status)](https://app.netlify.com/sites/endearing-stroopwafel-3da573/deploys)

## Features

- 🎨 **Beautiful Glass-morphism Design**
  - Modern, translucent UI elements
  - Animated background bubbles
  - Smooth transitions and animations
  - Responsive layout

- 🔐 **User Authentication**
  - Email/password registration and login
  - Secure session management
  - Protected routes
  - User profiles

- 📝 **Post Management**
  - Create, read, update, and delete posts
  - Rich text content
  - Post voting system
  - Comment threads
  - Real-time updates

- ⚡ **Performance**
  - Virtualized list for efficient post rendering
  - Infinite scrolling
  - Optimized database queries
  - Lazy loading of components

- 🔒 **Security**
  - Row Level Security (RLS) policies
  - Protected API endpoints
  - Secure user sessions
  - Input validation and sanitization

## Tech Stack

- **Frontend**
  - React 18
  - TypeScript
  - Tailwind CSS
  - Framer Motion
  - React Router
  - React Virtualized
  - Lucide Icons

- **Backend**
  - Supabase
  - PostgreSQL
  - Row Level Security

- **Testing**
  - Playwright
  - End-to-end testing
  - Component testing

- **Deployment**
  - Netlify
  - Continuous Deployment
  - SSL/TLS encryption

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn
- Supabase account

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/community-forum.git
   cd community-forum
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

1. Create a new Supabase project
2. Run the migration files in the `supabase/migrations` directory
3. Enable Row Level Security (RLS) policies

### Testing

Run the test suite:
```bash
npm run test
```

Run tests with UI:
```bash
npm run test:ui
```

### Deployment

The project is deployed on Netlify with continuous deployment enabled. Any changes pushed to the main branch will trigger a new deployment automatically.

To deploy your own instance:

1. Fork this repository
2. Create a new site on Netlify
3. Connect your forked repository
4. Set up the environment variables:
   - `VITE_SUPABASE_URL`
   - `VITE_SUPABASE_ANON_KEY`
5. Deploy!

## Project Structure

```
community-forum/
├── src/
│   ├── components/     # Reusable UI components
│   ├── contexts/       # React contexts
│   ├── lib/           # Utility functions and configurations
│   ├── pages/         # Page components
│   └── types/         # TypeScript type definitions
├── tests/             # Playwright tests
├── supabase/          # Supabase configurations and migrations
└── public/            # Static assets
```

## Database Schema

### Tables

- **profiles**
  - `id` (uuid, PK)
  - `username` (text, unique)
  - `created_at` (timestamp)
  - `updated_at` (timestamp)

- **posts**
  - `id` (uuid, PK)
  - `title` (text)
  - `content` (text)
  - `author_id` (uuid, FK)
  - `created_at` (timestamp)
  - `updated_at` (timestamp)

- **comments**
  - `id` (uuid, PK)
  - `content` (text)
  - `post_id` (uuid, FK)
  - `author_id` (uuid, FK)
  - `created_at` (timestamp)

- **votes**
  - `id` (uuid, PK)
  - `post_id` (uuid, FK)
  - `user_id` (uuid, FK)
  - `value` (integer)
  - `created_at` (timestamp)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Supabase](https://supabase.io/) for the amazing backend platform
- [Tailwind CSS](https://tailwindcss.com/) for the utility-first CSS framework
- [Framer Motion](https://www.framer.com/motion/) for the smooth animations
- [Lucide](https://lucide.dev/) for the beautiful icons
- [Netlify](https://www.netlify.com/) for hosting and continuous deployment