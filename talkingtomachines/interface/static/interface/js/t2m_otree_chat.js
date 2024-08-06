document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('t2m__chat-box'),
        userInput = document.getElementById('t2m__chat-human-input'),
        btnSend = document.getElementById('t2m__chat-btn-send');

    function sendMessage() {
        const message = userInput.value.trim();
        if (message === "") return;

        // Display user's message
        displayMessage(message, 't2m__chat-human-message');

        // Clear input
        userInput.value = "";

        liveSend(message);
    }

    window.liveRecv = (data) => {
        displayMessage(data, 't2m__chat-ai-message');
    }

    function displayMessage(message, className) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('t2m__chat-message', className);
        messageElement.innerHTML = `<div>${message}</div>`;
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    btnSend.addEventListener('click', sendMessage);

    js_vars.thread.forEach(
        t => {
            displayMessage(t.prompt, 't2m__chat-human-message');
            displayMessage(t.response, 't2m__chat-ai-message');
        }
    )
})