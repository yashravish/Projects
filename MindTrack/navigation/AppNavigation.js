// src/navigation/AppNavigator.js
import React from 'react';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import HomeScreen from '../screens/HomeScreen';
import MoodTrackerScreen from '../screens/MoodTrackerScreen';
import JournalScreen from '../screens/JournalScreen';
import MeditationScreen from '../screens/MeditationScreen';
import CommunityScreen from '../screens/CommunityScreen';
import ProgressScreen from '../screens/ProgressScreen';
import EmergencyScreen from '../screens/EmergencyScreen';
import DirectoryScreen from '../screens/DirectoryScreen';

const Stack = createNativeStackNavigator();

export default function AppNavigator() {
  return (
    <Stack.Navigator initialRouteName="Home">
      <Stack.Screen name="Home" component={HomeScreen} />
      <Stack.Screen name="MoodTracker" component={MoodTrackerScreen} />
      <Stack.Screen name="Journal" component={JournalScreen} />
      <Stack.Screen name="Meditation" component={MeditationScreen} />
      <Stack.Screen name="Community" component={CommunityScreen} />
      <Stack.Screen name="Progress" component={ProgressScreen} />
      <Stack.Screen name="Emergency" component={EmergencyScreen} />
      <Stack.Screen name="Directory" component={DirectoryScreen} />
    </Stack.Navigator>
  );
}
