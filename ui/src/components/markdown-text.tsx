import React from "react";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";
import rehypeSanitize from "rehype-sanitize";
import remarkGfm from "remark-gfm";

// Code highlighting with shiki
import { Highlighter } from "shiki";

import { formatMarkdown } from "@/lib";

interface MarkdownTextProps {
  content: string;
  className?: string;
}

const MarkdownText: React.FC<MarkdownTextProps> = ({ content, className }) => {
  const formattedContent = formatMarkdown(content);

  return (
    <ReactMarkdown
      className={`markdown ${className || ""}`}
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeRaw, rehypeSanitize]}
      components={{
        code({ className, children, ...props }: any) {
          const match = /language-(\w+)/.exec(className || "");
          const isInline = !match;
          
          if (isInline) {
            return (
              <code className="bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded" {...props}>
                {children}
              </code>
            );
          }

          return (
            <div className="relative">
              <pre className={`${match ? `language-${match[1]}` : ""} rounded-md`}>
                <code className={match ? `language-${match[1]}` : ""} {...props}>
                  {children}
                </code>
              </pre>
            </div>
          );
        },
        table({ children }: any) {
          return (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse">{children}</table>
            </div>
          );
        },
        // Other custom components can be added here
      }}
    >
      {formattedContent}
    </ReactMarkdown>
  );
};

export default MarkdownText;