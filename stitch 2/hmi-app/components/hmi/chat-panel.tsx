"use client";

import { useMutation } from "@tanstack/react-query";
import { useQueryClient } from "@tanstack/react-query";
import { FormEvent, useState, useRef, useEffect } from "react";
import { sendChatMessage } from "@/lib/api";
import { useHmiStore } from "@/store/hmi-store";
import { Send } from "lucide-react";
import type { HmiSnapshot } from "@/lib/types";

const ALLOWED_TAGS = new Set(["p", "ul", "ol", "li", "strong", "em", "br", "code", "pre", "a", "span"]);
const STRIP_TAGS = new Set(["script", "style", "iframe", "object", "embed", "link", "meta"]);

function escapeHtml(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function sanitizeAssistantHtml(input: string): string {
  const parser = new DOMParser();
  const doc = parser.parseFromString(`<div>${input}</div>`, "text/html");
  const root = doc.body.firstElementChild;

  if (!root) return `<p>${escapeHtml(input)}</p>`;

  const sanitizeNode = (node: Node): string => {
    if (node.nodeType === Node.TEXT_NODE) {
      return escapeHtml(node.textContent ?? "");
    }

    if (node.nodeType !== Node.ELEMENT_NODE) return "";

    const el = node as Element;
    const tag = el.tagName.toLowerCase();

    if (STRIP_TAGS.has(tag)) return "";

    const children = Array.from(el.childNodes).map(sanitizeNode).join("");

    if (!ALLOWED_TAGS.has(tag)) return children;

    if (tag === "a") {
      const href = (el.getAttribute("href") ?? "").trim();
      const safeHref =
        href.startsWith("http://") || href.startsWith("https://") || href.startsWith("/") ? href : "";
      if (!safeHref) return children;
      return `<a href="${escapeHtml(safeHref)}" target="_blank" rel="noopener noreferrer">${children}</a>`;
    }

    if (tag === "br") return "<br/>";

    return `<${tag}>${children}</${tag}>`;
  };

  const sanitized = Array.from(root.childNodes).map(sanitizeNode).join("").trim();
  return sanitized || `<p>${escapeHtml(input)}</p>`;
}

export default function ChatPanel() {
  const { sessionId, messages, addLocalUserMessage, addMessage } = useHmiStore();
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: sendChatMessage,
    onSuccess: (data) => addMessage(data)
  });

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!input.trim() || mutation.isPending) return;
    const text = input.trim();
    const currentSnapshot = queryClient.getQueryData<HmiSnapshot>(["hmi-snapshot"]) ?? null;
    addLocalUserMessage(text);
    setInput("");
    mutation.mutate({ sessionId, message: text, currentSnapshot });
  }

  return (
    <div className="flex h-full flex-col bg-[#0a0f1a]">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-[#0d1320] scrollbar-hide [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center px-4">
            <p className="text-sm text-slate-500">
              Ask about turbine health, decay predictions, or maintenance recommendations.
            </p>
          </div>
        ) : (
          messages.map((m) => (
            <div key={m.id} className="space-y-1">
              {m.role === "user" ? (
                // User message - pill style like VS Code
                <div className="flex justify-end">
                  <div className="inline-block rounded-2xl bg-primary/20 px-4 py-2 text-sm text-slate-100 max-w-[85%]">
                    {m.content}
                  </div>
                </div>
              ) : (
                // Assistant message - bullet point style
                <div className="flex gap-3">
                  <div className="mt-2 h-2 w-2 rounded-full bg-slate-400 flex-shrink-0" />
                  <div
                    className="flex-1 text-sm text-slate-200 leading-relaxed [&_p]:mb-2 [&_ul]:mb-2 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:mb-2 [&_ol]:list-decimal [&_ol]:pl-5 [&_li]:mb-1 [&_code]:rounded [&_code]:bg-slate-800 [&_code]:px-1 [&_pre]:mb-2 [&_pre]:overflow-x-auto [&_pre]:rounded-md [&_pre]:bg-slate-900 [&_pre]:p-2"
                    dangerouslySetInnerHTML={{ __html: sanitizeAssistantHtml(m.content) }}
                  >
                  </div>
                </div>
              )}
            </div>
          ))
        )}
        {mutation.isPending && (
          <div className="flex gap-3">
            <div className="mt-2 h-2 w-2 rounded-full bg-slate-400 flex-shrink-0 animate-pulse" />
            <div className="text-sm text-slate-400">Thinking...</div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-slate-700 bg-[#151c2c] p-4">
        <form onSubmit={onSubmit} className="relative">
          <input
            className="w-full rounded-xl border border-slate-600 bg-[#0a0f1a] px-4 py-3 pr-12 text-sm text-slate-100 placeholder-slate-400 outline-none transition-colors focus:border-primary/50 focus:ring-1 focus:ring-primary/30"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question..."
          />
          <button
            type="submit"
            disabled={mutation.isPending || !input.trim()}
            className="absolute right-2 top-1/2 -translate-y-1/2 rounded-lg bg-primary p-2 text-white transition-opacity hover:opacity-80 disabled:opacity-40"
          >
            <Send className="h-4 w-4" />
          </button>
        </form>
        {mutation.error && (
          <p className="mt-2 text-xs text-red-400">
            Chat failed. Check API keys and backend.
          </p>
        )}
      </div>
    </div>
  );
}
