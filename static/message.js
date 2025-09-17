import { requestAndPlayAnimation } from './avatar.js';

export async function sendMessage() {
    const input = document.getElementById("message");
    const user_name = document.getElementById("user_name");
    const language = document.getElementById('language');
    const motion = document.getElementById("motion");
    const emojiButton = document.getElementById("emoji-btn");
    const chat = document.getElementById("chat");

    const userMessage = input.value.trim();
    if (!userMessage) return;

    const userName = user_name.value.trim();
    if (!userName) return;

    chat.innerHTML += `<div><span class="user-name">${userName} ${emojiButton.textContent}:</span> ${userMessage}</div>`;

    const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            language: language.value,
            my_name_is: userName,
            answer: userMessage,
            emotion: emojiButton.textContent,
            thought: "",
            motion: "",
            association_words: [""]
        })
    });

    const data = await response.json();
    if (data) {
        chat.innerHTML += `<div><span class="user-name">${data.my_name_is} ${data.emotion}:</span> ${data.answer}</div>`;
    } else {
        chat.innerHTML += `<div><span class="user-name">server:</span> ${data.error}</div>`;
    }

    if (data.motion.length >= 3)
        requestAndPlayAnimation(data.motion);
    motion.innerHTML = data.motion;

    input.value = "";
    chat.scrollTop = chat.scrollHeight;
}

window.sendMessage = sendMessage;