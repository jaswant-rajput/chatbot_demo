import { useState, useEffect, useRef } from "react";
import "./App.css"; // Assuming styles are in App.css or inline them if preferred
import crossIcon from "./assets/cross.png";
function App() {
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [messages, setMessages] = useState<{ text: string; sender: string }[]>(
    [],
  );
  const [input, setInput] = useState("");
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
        <div className="chat-window">
          <div className="chat-window-header">
            <div className="chatbot-logo">🤖</div>
            <div className="chatbot-title">Health AI</div>
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
                    message.sender === "user" ? "#007bff" : "#f1f0f0",
                  color: message.sender === "user" ? "white" : "black",
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
            <input
              type="text"
              className="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
            />
            <button className="send-button" onClick={sendMessage}>
              Send
            </button>
          </div>
        </div>
      )}
    </>
  );
}

export default App;
