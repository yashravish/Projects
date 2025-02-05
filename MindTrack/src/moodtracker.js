// src/components/MoodTracker.js
import React, { useState, useContext } from 'react';
import { View, Text, Pressable } from 'react-native';
import { WellnessContext } from '../context/WellnessContext';

export default function MoodTracker() {
  const { addMoodEntry } = useContext(WellnessContext);
  const [selectedMood, setSelectedMood] = useState(null);

  const moods = ['😃', '🙂', '😐', '🙁', '😢'];

  const handleMoodSelect = (mood) => {
    setSelectedMood(mood);
    addMoodEntry({ date: new Date(), mood });
  };

  return (
    <View className="p-4">
      <Text className="text-xl font-bold mb-4">How are you feeling today?</Text>
      <View className="flex-row justify-around">
        {moods.map((mood, index) => (
          <Pressable
            key={index}
            onPress={() => handleMoodSelect(mood)}
            className={`p-4 rounded-full border ${selectedMood === mood ? 'bg-blue-200' : 'bg-gray-200'}`}
          >
            <Text className="text-2xl">{mood}</Text>
          </Pressable>
        ))}
      </View>
      {selectedMood && (
        <Text className="mt-4 text-center">You selected: {selectedMood}</Text>
      )}
    </View>
  );
}
