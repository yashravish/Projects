// src/components/MeditationPlayer.js
import React from 'react';
import { View, Text, Pressable } from 'react-native';

export default function MeditationPlayer() {
  return (
    <View className="p-4 items-center">
      <Text className="text-xl font-bold mb-4">Guided Meditation</Text>
      <Text className="mb-4">Listen to our guided meditation to relax and center your mind.</Text>
      <Pressable className="bg-purple-500 py-2 px-4 rounded">
        <Text className="text-white">Play Meditation</Text>
      </Pressable>
    </View>
  );
}
