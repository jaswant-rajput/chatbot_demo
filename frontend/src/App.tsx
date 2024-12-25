import { useState, useEffect, useRef } from "react";
import "./App.css"; // Assuming styles are in App.css or inline them if preferred
import crossIcon from "./assets/cross.png";
import imagePng from "./assets/logo.png";
function App() {
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [messages, setMessages] = useState<{ text: string; sender: string }[]>(
    [],
  );
  const [input, setInput] = useState("");
  const [disableButton, setDisableButton] = useState(false);
  const chatBodyRef = useRef<HTMLDivElement | null>(null);

  const toggleChatWindow = () => {
    setIsChatOpen((prev) => !prev);
  };

  useEffect(() => {
    if (chatBodyRef.current) {
      chatBodyRef.current.scrollTop = chatBodyRef.current.scrollHeight;
    }
  }, [messages]);

  const sendMessage = async () => {
    setDisableButton(true);
    if (input.trim() === "") return;

    const userMessage = { text: input, sender: "user" };
    setMessages([...messages, userMessage]);
    setInput("");

    try {
      const response = await fetch("http://127.0.0.1:8000/send_message", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: [{ sender: "user", message: input }],
          message_id: 12345,
          host_url: "https://example.com/api/chat",
          prompt: "What are your opening hours?",
          pinecone_index: "drmalpani",
          namespace: "ivfindia",
          closure_msg: "Is there anything else I can assist you with?",
          conversation_status: "active",
          org_description:
            "We are a customer support service for an e-commerce platform.",
          unsure_msg:
            "I'm sorry, I'm not sure how to respond to that. Can you please rephrase?",
          sender_country: "USA",
          sender_city: "San Francisco",
          filters: {
            category: "support",
            priority: "high",
          },
          buckets: [],
        }),
      });

      const data = await response.json();

      if (data.status === 200 && data.data.ai_response) {
        await streamAIResponse(data.data.ai_response[0]);
      } else {
        setMessages((prevMessages) => [
          ...prevMessages,
          {
            text: "Sorry, something went wrong. Please try again.",
            sender: "bot",
          },
        ]);
      }
    } catch (error) {
      console.error("Error fetching chatbot response:", error);
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          text: "Sorry, there was an error connecting to the server.",
          sender: "bot",
        },
      ]);
    }
    setDisableButton(false);
  };

  const streamAIResponse = async (responseText: string) => {
    let accumulatedText = "";
    for (const char of responseText) {
      accumulatedText += char;
      await new Promise((resolve) => setTimeout(resolve, 30)); // Simulates streaming delay
      setMessages((prevMessages) => {
        const updatedMessages = [...prevMessages];
        const lastMessage = updatedMessages[updatedMessages.length - 1];
        if (lastMessage?.sender === "bot") {
          updatedMessages[updatedMessages.length - 1] = {
            ...lastMessage,
            text: accumulatedText,
          };
        } else {
          updatedMessages.push({ text: accumulatedText, sender: "bot" });
        }
        return updatedMessages;
      });
    }
  };
  return (
    <>
      <div className="chatbot-button" onClick={toggleChatWindow}>
        💬
      </div>

      {isChatOpen && (
        <div
          className="chat-window"
          style={{
            backgroundColor: "rgb(3 22 52 / 1)",
          }}
        >
          <div className="chat-window-header">
            <img
              className="h-full w-full bg-token-main-surface-secondary"
              alt="GPT Icon"
              src={imagePng}
            />
            <div className="chatbot-title" style={{ fontFamily: "sans-serif" }}>
              Health AI
            </div>
            <img
              src={crossIcon}
              className="close-button"
              onClick={toggleChatWindow}
              alt="close"
            />
          </div>
          <div className="chat-window-body" ref={chatBodyRef}>
            {messages.map((message, index) => (
              <div
                key={index}
                className={`message ${
                  message.sender === "user" ? "user-message" : "bot-message"
                }`}
                style={{
                  alignSelf:
                    message.sender === "user" ? "flex-end" : "flex-start",
                  backgroundColor:
                    message.sender === "user" ? "rgb(29 36 57 / 1)" : "",
                  lineHeight: "1.435rem",
                  fontSize: "0.9375rem",
                  fontFamily: "sans-serif",
                  color:
                    message.sender === "user"
                      ? "rgb(242 221 204 / 1)"
                      : "rgb(242 221 204 / 1)",
                  padding: "10px",
                  borderRadius: "10px",
                  margin: "5px 0",
                  maxWidth: "70%",
                }}
              >
                {message.text}
              </div>
            ))}
          </div>
          <div className="chat-window-footer">
            <div className="chat-window-footer-div">
              <input
                type="text"
                className="chat-input"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type your message..."
                onKeyDown={(e) => {
                  if (e.key == "Enter" && !disableButton) {
                    sendMessage();
                  }
                }}
              />
              <div className="relative-container">
                <div className="opacity-style">
                  <button
                    type="button"
                    className="submit-button dark-theme"
                    title="Submit message"
                    aria-label="Submit message"
                    disabled={disableButton}
                    onClick={sendMessage}
                  >
                    <svg
                      viewBox="0 0 24 24"
                      fill="currentColor"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path d="M4.20889 10.7327C3.9232 11.0326 3.93475 11.5074 4.23467 11.7931C4.5346 12.0788 5.00933 12.0672 5.29502 11.7673L11.2495 5.516V20.25C11.2495 20.6642 11.5853 21 11.9995 21C12.4137 21 12.7495 20.6642 12.7495 20.25V5.51565L18.7043 11.7673C18.99 12.0672 19.4648 12.0788 19.7647 11.7931C20.0646 11.5074 20.0762 11.0326 19.7905 10.7327L12.7238 3.31379C12.5627 3.14474 12.3573 3.04477 12.1438 3.01386C12.0971 3.00477 12.0489 3 11.9995 3C11.9498 3 11.9012 3.00483 11.8543 3.01406C11.6412 3.04518 11.4363 3.14509 11.2756 3.31379L4.20889 10.7327Z"></path>
                    </svg>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default App;
