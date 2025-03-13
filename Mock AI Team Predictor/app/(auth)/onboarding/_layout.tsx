import { Stack } from 'expo-router';

export default function OnboardingLayout() {
  return (
    <Stack
      screenOptions={{
        headerShown: false,
        animation: 'slide_from_right',
      }}
    >
      <Stack.Screen name="index" />
      <Stack.Screen name="skills" />
      <Stack.Screen name="personality" />
      <Stack.Screen name="goals" />
      <Stack.Screen name="analyzing" />
    </Stack>
  );
}