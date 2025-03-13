# AI Team Predictor

An intelligent team matching platform that uses AI to connect professionals based on their skills, work style, and career goals. Built with React Native and Expo.

![AI Team Predictor](https://images.unsplash.com/photo-1522071820081-009f0129c71c?q=80&w=2940&auto=format&fit=crop)

## Features

- **AI-Powered Matching**: Advanced algorithms to find your perfect team match
- **Skill Assessment**: Comprehensive skill tracking and evaluation
- **Work Style Analysis**: Personality and work preference matching
- **Career Goals Alignment**: Long-term and short-term goal compatibility
- **Real-time Chat**: Direct communication with potential team members
- **Smart Analytics**: AI-driven compatibility scores and insights

## Tech Stack

- **Framework**: React Native + Expo
- **Navigation**: Expo Router
- **Database**: Supabase
- **Authentication**: Supabase Auth
- **State Management**: React Context + Hooks
- **UI Components**: Custom components with React Native core
- **Animations**: React Native Reanimated
- **Gestures**: React Native Gesture Handler
- **Icons**: Lucide React Native
- **Fonts**: Google Fonts (Inter)

## Getting Started

### Prerequisites

- Node.js 18 or higher
- npm or yarn
- Expo CLI

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-team-predictor.git
```

2. Install dependencies:
```bash
npm install
```

3. Set up environment variables:
Create a `.env` file in the root directory with the following variables:
```
EXPO_PUBLIC_SUPABASE_URL=your_supabase_url
EXPO_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
```

4. Start the development server:
```bash
npm run dev
```

## Project Structure

```
ai-team-predictor/
├── app/                    # Application routes
│   ├── (auth)/            # Authentication routes
│   │   └── onboarding/    # User onboarding flow
│   └── (tabs)/            # Main app tabs
├── components/            # Reusable components
├── hooks/                 # Custom React hooks
├── types/                 # TypeScript type definitions
└── assets/               # Static assets
```

### Key Directories

- **/app**: Contains all the routes and screens using Expo Router
- **/components**: Reusable UI components
- **/hooks**: Custom React hooks for shared logic
- **/types**: TypeScript type definitions and interfaces

## Features in Detail

### Authentication & Onboarding

- Email-based authentication
- Multi-step onboarding process:
  - Basic information collection
  - Skill assessment
  - Work style evaluation
  - Career goals alignment

### Main Features

1. **Dashboard**
   - AI match score
   - Potential matches count
   - Team rating
   - Top skills overview

2. **Matches**
   - AI-powered team recommendations
   - Compatibility percentage
   - Skill alignment visualization
   - Quick connect options

3. **Chat**
   - Real-time messaging
   - Conversation history
   - Unread message indicators
   - Online status

4. **Profile**
   - Skill management
   - Work preferences
   - Career goals
   - Team preferences

## Development Guidelines

### Code Style

- Use TypeScript for type safety
- Follow React Native best practices
- Implement proper error handling
- Write meaningful comments
- Use consistent naming conventions

### Component Structure

```typescript
import { StyleSheet } from 'react-native';

interface Props {
  // Component props
}

export function Component({ prop1, prop2 }: Props) {
  // Component logic
  return (
    // JSX
  );
}

const styles = StyleSheet.create({
  // Styles
});
```

### State Management

- Use React Context for global state
- Implement hooks for shared logic
- Keep component state local when possible

### Navigation

- Use Expo Router for type-safe navigation
- Implement proper navigation patterns
- Handle deep linking appropriately

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Expo](https://expo.dev/) for the amazing development platform
- [Supabase](https://supabase.com/) for the backend infrastructure
- [Unsplash](https://unsplash.com/) for the beautiful images