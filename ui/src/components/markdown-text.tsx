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

  // In react-markdown v10, className is passed to wrapper div instead
  return (
    <div className={`markdown ${className || ""}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw, rehypeSanitize]}
        components={{
          code({ className, children, ...props }: any) {
            const match = /language-(\w+)/.exec(className || "");
            const isInline = !match;
            
            if (isInline) {
              return (
                <code style={{backgroundColor: '#f3f4f6', padding: '0 0.25rem', borderRadius: '0.25rem'}} {...props}>
                  {children}
                </code>
              );
            }

            return (
              <div className="relative">
                <pre style={{borderRadius: '0.375rem'}}>
                  <code className={match ? `language-${match[1]}` : ""} {...props}>
                    {children}
                  </code>
                </pre>
              </div>
            );
          },
          table({ children }: any) {
            return (
              <div style={{overflowX: 'auto'}}>
                <table style={{width: '100%', borderCollapse: 'collapse'}}>{children}</table>
              </div>
            );
          },
          // Other custom components can be added here
        }}
      >
        {formattedContent}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownText;