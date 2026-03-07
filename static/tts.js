// static/js/tts.js

function readContent(elementId) {
    const element = document.getElementById(elementId);
    if (!element) return;

    const text = element.innerText;
    if (!text) return;

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1;
    utterance.pitch = 1;
    utterance.volume = 1;

    // Stop any ongoing speech
    window.speechSynthesis.cancel();

    window.speechSynthesis.speak(utterance);
}