// Use native fetch (Node 18+)
async function testSignup() {
    console.log('Attempting signup request to http://127.0.0.1:3001/v1/auth/signup...');
    try {
        const response = await fetch('http://127.0.0.1:3001/v1/auth/signup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                email: 'debug_user_' + Date.now() + '@example.com',
                password: 'Password123!'
            })
        });

        console.log('Response Status:', response.status);
        const text = await response.text();
        console.log('Response Body:', text);
    } catch (e: any) {
        console.error('Fetch failed:', e.message);
        if (e.cause) console.error('Cause:', e.cause);
    }
}
testSignup();
