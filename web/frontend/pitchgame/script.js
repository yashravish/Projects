// script.js

// Wait for the DOM to load
document.addEventListener('DOMContentLoaded', () => {
    // List of musical notes
    const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

    // Game variables
    let currentNote = '';
    let attempt = 1;
    let score = 0;
    let currentAudio = null;  // Variable to store the currently playing audio

    // DOM elements
    const startScreen = document.getElementById('start-screen');
    const gameScreen = document.getElementById('game-screen');
    const gameOverScreen = document.getElementById('game-over-screen');
    const startButton = document.getElementById('start-game');
    const resetButton = document.getElementById('reset-game');
    const playAgainButton = document.getElementById('play-again');
    const attemptDisplay = document.getElementById('attempt');
    const scoreDisplay = document.getElementById('score');
    const finalScoreDisplay = document.getElementById('final-score');
    const notesGrid = document.querySelector('.grid');

    /**
     * Function to initialize the game
     */
    function initGame() {
        // Reset variables
        attempt = 1;
        score = 0;
        scoreDisplay.textContent = `Score: ${score}/10`;
        attemptDisplay.textContent = `Question ${attempt}/10`;

        // Hide other screens
        startScreen.classList.add('hidden');
        gameOverScreen.classList.add('hidden');
        gameScreen.classList.remove('hidden');

        // Generate note buttons
        generateNoteButtons();

        // Start the first question
        playRandomNote();
    }

    /**
     * Function to generate note buttons
     */
    function generateNoteButtons() {
        // Clear existing buttons
        notesGrid.innerHTML = '';

        // Create buttons for each note
        notes.forEach(note => {
            const button = document.createElement('button');
            button.textContent = note;
            button.className = 'bg-gray-200 text-black px-4 py-2 rounded hover:bg-gray-300 transition duration-300';
            button.addEventListener('click', () => handleNoteSelection(note, button));
            notesGrid.appendChild(button);
        });
    }

    /**
     * Function to play a random note
     */
    function playRandomNote() {
        // Stop any currently playing note
        if (currentAudio) {
            currentAudio.pause();
            currentAudio.currentTime = 0;
        }

        // Select a new random note
        currentNote = notes[Math.floor(Math.random() * notes.length)];

        // Load and play the audio file for the note
        currentAudio = new Audio(`audio/notes/${currentNote}.mp3`);
        currentAudio.play().catch(error => {
            console.error('Error playing note audio:', error);
        });
    }

    /**
     * Function to handle note selection
     * @param {string} selectedNote - The note selected by the user
     * @param {HTMLElement} button - The button element clicked
     */
    function handleNoteSelection(selectedNote, button) {
        // Disable buttons temporarily
        disableNoteButtons(true);

        // Check if the selected note is correct
        if (selectedNote === currentNote) {
            // Correct answer
            score++;
            scoreDisplay.textContent = `Score: ${score}/10`;

            // Stop and play success sound
            stopCurrentAudio();
            currentAudio = new Audio('audio/success.mp3');
            currentAudio.play();

            // Add animation for correct answer
            button.classList.add('correct-answer');
        } else {
            // Incorrect answer
            // Stop and play failure sound
            stopCurrentAudio();
            currentAudio = new Audio('audio/failure.mp3');
            currentAudio.play();

            // Add animation for incorrect answer
            button.classList.add('incorrect-answer');
        }

        // Wait before moving to the next question
        setTimeout(() => {
            // Reset button styles
            button.classList.remove('correct-answer', 'incorrect-answer');

            // Enable buttons
            disableNoteButtons(false);

            // Move to the next question or end game
            attempt++;
            if (attempt <= 10) {
                attemptDisplay.textContent = `Question ${attempt}/10`;
                playRandomNote();
            } else {
                endGame();
            }
        }, 1000);
    }

    /**
     * Function to stop any currently playing audio
     */
    function stopCurrentAudio() {
        if (currentAudio) {
            currentAudio.pause();
            currentAudio.currentTime = 0;
        }
    }

    /**
     * Function to disable or enable note buttons
     * @param {boolean} disable - True to disable, false to enable
     */
    function disableNoteButtons(disable) {
        const buttons = notesGrid.querySelectorAll('button');
        buttons.forEach(button => {
            button.disabled = disable;
        });
    }

    /**
     * Function to end the game
     */
    function endGame() {
        // Hide game screen and show game over screen
        gameScreen.classList.add('hidden');
        gameOverScreen.classList.remove('hidden');

        // Display final score
        finalScoreDisplay.textContent = `Your Score: ${score}/10`;
    }

    /**
     * Function to reset the game
     */
    function resetGame() {
        // Reset variables and UI elements
        attempt = 1;
        score = 0;
        scoreDisplay.textContent = `Score: ${score}/10`;
        attemptDisplay.textContent = `Question ${attempt}/10`;

        // Re-enable buttons
        disableNoteButtons(false);

        // Start over
        playRandomNote();
    }

    // Event listeners
    startButton.addEventListener('click', initGame);
    resetButton.addEventListener('click', resetGame);
    playAgainButton.addEventListener('click', initGame);
});

