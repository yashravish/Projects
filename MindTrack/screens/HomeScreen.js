// src/screens/HomeScreen.js
import React from 'react';
import { View, Text, Pressable } from 'react-native';
import { useNavigation } from '@react-navigation/native';

export default function HomeScreen() {
  const navigation = useNavigation();

  return (
    <View className="flex-1 items-center justify-center bg-white p-4">
      <Text className="text-2xl font-bold mb-4">Mental Wellness App</Text>
      <Pressable
        onPress={() => navigation.navigate('MoodTracker')}
        className="bg-blue-500 py-2 px-4 rounded mb-2"
      >
        <Text className="text-white">Mood Tracker</Text>
      </Pressable>
      <Pressable
        onPress={() => navigation.navigate('Journal')}
        className="bg-green-500 py-2 px-4 rounded mb-2"
      >
        <Text className="text-white">Daily Journal</Text>
      </Pressable>
      <Pressable
        onPress={() => navigation.navigate('Meditation')}
        className="bg-purple-500 py-2 px-4 rounded mb-2"
      >
        <Text className="text-white">Meditation</Text>
      </Pressable>
      <Pressable
        onPress={() => navigation.navigate('Community')}
        className="bg-yellow-500 py-2 px-4 rounded mb-2"
      >
        <Text className="text-white">Community</Text>
      </Pressable>
      <Pressable
        onPress={() => navigation.navigate('Progress')}
        className="bg-indigo-500 py-2 px-4 rounded mb-2"
      >
        <Text className="text-white">Progress</Text>
      </Pressable>
      <Pressable
        onPress={() => navigation.navigate('Emergency')}
        className="bg-red-500 py-2 px-4 rounded mb-2"
      >
        <Text className="text-white">Emergency Resources</Text>
      </Pressable>
      <Pressable
        onPress={() => navigation.navigate('Directory')}
        className="bg-gray-500 py-2 px-4 rounded"
      >
        <Text className="text-white">Professional Directory</Text>
      </Pressable>
    </View>
  );
}
