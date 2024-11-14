document.addEventListener('DOMContentLoaded', function () {
    const display = document.querySelector('.display');
    const buttons = document.querySelectorAll('.btn');
    const clearButton = Array.from(buttons).find(btn => btn.textContent === 'C'); // Find the 'C' button

    let currentInput = '';
    let operator = '';
    let previousInput = '';
    let resultShown = false;

    function updateDisplay(value) {
        if (value === '0') {
            display.textContent = value; // Display simple 0 after reset
        } else if (Math.abs(value) >= 1e8 || (Math.abs(value) < 1e-8 && value !== 0)) {
            display.textContent = Number(value).toExponential(5); // Scientific notation for large or small values
        } else {
            display.textContent = value.toString().slice(0, 12); // Prevent overflow on the calculator screen
        }
    }

    function handleInput(value) {
        if (value === 'C') {
            resetCalculator();
            return; // Always reset calculator on 'C'
        }

        if (resultShown) {
            if (!isNaN(value) || value === '.') {
                currentInput = '';
                resultShown = false;
            }
        }

        if (!isNaN(value) || value === '.') {
            currentInput += value;
            updateDisplay(currentInput);
        } else if (value === '=') {
            if (operator && previousInput && currentInput) {
                try {
                    currentInput = eval(`${previousInput} ${operator} ${currentInput}`);
                    updateDisplay(currentInput);
                    operator = '';
                    previousInput = '';
                    resultShown = true;
                } catch {
                    updateDisplay('Error');
                    resetCalculator();
                }
            }
        } else if (['+', '-', '*', '/'].includes(value)) {
            operator = value;
            previousInput = currentInput;
            currentInput = '';
        }
    }

    function resetCalculator() {
        currentInput = '';
        operator = '';
        previousInput = '';
        updateDisplay('0'); // Display simple 0 after reset
        resultShown = false;
    }

    buttons.forEach(button => {
        button.addEventListener('click', () => {
            handleInput(button.textContent);
        });
    });

    document.addEventListener('keydown', (event) => {
        const key = event.key;
        if (key.toUpperCase() === 'C') { 
            clearButton.click(); // Simulate a click on the 'C' button
        } else if (!isNaN(key) || key === '.' || key === '=' || key === 'Enter' || ['+', '-', '*', '/'].includes(key)) {
            handleInput(key === 'Enter' ? '=' : key);
        } else if (key === 'Backspace') {
            if (resultShown) {
                resetCalculator();
            } else {
                currentInput = currentInput.slice(0, -1);
                updateDisplay(currentInput || '0');
            }
        }
    });
});



