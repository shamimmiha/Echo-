export type PricingType = "Free" | "Freemium" | "Paid" | "Free Trial" | "Contact for Pricing";

export type CategoryType = 
  | "Natural Language Processing" 
  | "Computer Vision" 
  | "Productivity" 
  | "Design" 
  | "Audio/Video" 
  | "Developer Tools"
  | "Business Intelligence" 
  | "Research" 
  | "Healthcare"
  | "Education";

export interface Tool {
  id: string;
  name: string;
  description: string;
  logo: string;
  website: string;
  pricing: PricingType;
  categories: CategoryType[];
  tags: string[];
  featured?: boolean;
  trending?: boolean;
}

export const tools: Tool[] = [
  {
    id: "1",
    name: "ChatGPT",
    description: "Conversational AI system that can engage in natural dialogues, answer questions, and assist with various tasks.",
    logo: "https://upload.wikimedia.org/wikipedia/commons/0/04/ChatGPT_logo.svg",
    website: "https://chat.openai.com",
    pricing: "Freemium",
    categories: ["Natural Language Processing", "Productivity"],
    tags: ["chatbot", "writing assistant", "language model"],
    featured: true,
    trending: true
  },
  {
    id: "2",
    name: "DALL-E",
    description: "AI system that can create realistic images and art from natural language descriptions.",
    logo: "https://upload.wikimedia.org/wikipedia/commons/a/a4/OpenAI_Logo.svg",
    website: "https://openai.com/dall-e-3",
    pricing: "Paid",
    categories: ["Design", "Computer Vision"],
    tags: ["image generation", "art", "creative"],
    featured: true
  },
  {
    id: "3",
    name: "Midjourney",
    description: "AI-powered tool that generates images from textual descriptions, focusing on artistic quality.",
    logo: "https://upload.wikimedia.org/wikipedia/commons/e/e6/Midjourney_Emblem.png",
    website: "https://www.midjourney.com",
    pricing: "Paid",
    categories: ["Design", "Computer Vision"],
    tags: ["image generation", "art", "creative"],
    trending: true
  },
  {
    id: "4",
    name: "GitHub Copilot",
    description: "AI pair programmer that helps you write better code by suggesting code completions in real-time.",
    logo: "https://github.githubassets.com/images/modules/site/copilot/copilot.png",
    website: "https://github.com/features/copilot",
    pricing: "Paid",
    categories: ["Developer Tools", "Productivity"],
    tags: ["coding assistant", "autocomplete", "programming"],
    featured: true
  },
  {
    id: "5",
    name: "Jasper",
    description: "AI content platform that helps teams create high-quality content faster.",
    logo: "https://www.jasper.ai/images/logos/jasper-primary.svg",
    website: "https://www.jasper.ai",
    pricing: "Paid",
    categories: ["Natural Language Processing", "Productivity"],
    tags: ["content creation", "marketing", "writing assistant"]
  },
  {
    id: "6",
    name: "Runway",
    description: "Creative suite with AI tools for video editing, generation, and visual effects.",
    logo: "https://storage.googleapis.com/runwayml-website-cms-prod/og_image_runway_d43f99ca90.jpg",
    website: "https://runwayml.com",
    pricing: "Freemium",
    categories: ["Audio/Video", "Design"],
    tags: ["video editing", "generative AI", "visual effects"],
    trending: true
  },
  {
    id: "7",
    name: "Hugging Face",
    description: "Platform providing tools for building, training and deploying machine learning models.",
    logo: "https://huggingface.co/front/assets/huggingface_logo.svg",
    website: "https://huggingface.co",
    pricing: "Freemium",
    categories: ["Developer Tools", "Natural Language Processing", "Research"],
    tags: ["machine learning", "models", "transformers"],
    featured: true
  },
  {
    id: "8",
    name: "Grammarly",
    description: "AI-powered writing assistant that helps with grammar, clarity, and style.",
    logo: "https://static.grammarly.com/assets/files/dbd7595425d7f5cf9b0d674d96d6849a/grammarly_logo.svg",
    website: "https://www.grammarly.com",
    pricing: "Freemium",
    categories: ["Natural Language Processing", "Productivity"],
    tags: ["writing assistant", "grammar", "proofreading"]
  },
  {
    id: "9",
    name: "Notion AI",
    description: "AI writing assistant integrated with Notion to help write, edit, and summarize content.",
    logo: "https://upload.wikimedia.org/wikipedia/commons/4/45/Notion_app_logo.png",
    website: "https://www.notion.so/product/ai",
    pricing: "Paid",
    categories: ["Productivity", "Natural Language Processing"],
    tags: ["writing assistant", "note-taking", "summarization"]
  },
  {
    id: "10",
    name: "Anthropic Claude",
    description: "Conversational AI assistant focused on helpfulness, harmlessness, and honesty.",
    logo: "https://upload.wikimedia.org/wikipedia/commons/2/20/Anthropic_Logo.png",
    website: "https://www.anthropic.com/claude",
    pricing: "Freemium",
    categories: ["Natural Language Processing"],
    tags: ["chatbot", "AI assistant", "language model"],
    trending: true
  },
  {
    id: "11",
    name: "Perplexity AI",
    description: "AI-powered search engine that provides comprehensive answers with cited sources.",
    logo: "https://assets-global.website-files.com/64f6c5724844bc1a217bc2c4/64f6c744aa41a47b86c6c53a_perplexity_logo__dark.svg",
    website: "https://www.perplexity.ai",
    pricing: "Freemium",
    categories: ["Natural Language Processing", "Research"],
    tags: ["search engine", "information retrieval", "answers"],
    trending: true
  },
  {
    id: "12",
    name: "Adept",
    description: "AI that can take actions in software applications based on natural language instructions.",
    logo: "https://adept.ai/images/logo-light.svg",
    website: "https://www.adept.ai",
    pricing: "Contact for Pricing",
    categories: ["Productivity", "Natural Language Processing"],
    tags: ["automation", "software agent", "action model"]
  },
  {
    id: "13",
    name: "Stable Diffusion",
    description: "Open-source AI art generator that creates detailed images from text descriptions using latent diffusion.",
    logo: "https://upload.wikimedia.org/wikipedia/commons/f/f8/Stable_Diffusion_logo.png",
    website: "https://stablediffusionweb.com",
    pricing: "Free",
    categories: ["Design", "Computer Vision"],
    tags: ["image generation", "open source", "text-to-image"],
    trending: true
  },
  {
    id: "14",
    name: "Gemini",
    description: "Google's multimodal AI model that can understand and generate text, code, images, and reason across different inputs.",
    logo: "https://storage.googleapis.com/gweb-uniblog-publish-prod/images/Gemini_Thumbnail.width-1300.format-webp.webp",
    website: "https://gemini.google.com",
    pricing: "Freemium",
    categories: ["Natural Language Processing", "Computer Vision", "Developer Tools"],
    tags: ["multimodal", "language model", "code assistant"],
    featured: true,
    trending: true
  },
  {
    id: "15",
    name: "Synthesia",
    description: "AI video generation platform that creates professional videos with virtual presenters from text.",
    logo: "https://assets-global.website-files.com/61dc0796f359b6145db3270e/62b41a2a4b43d95dba056d2b_Meta-Synthesia-Logo.png",
    website: "https://www.synthesia.io",
    pricing: "Paid",
    categories: ["Audio/Video", "Design"],
    tags: ["ai video", "virtual presenter", "video generation"]
  },
  {
    id: "16",
    name: "Bard",
    description: "AI chatbot by Google with real-time information access to provide helpful, accurate responses.",
    logo: "https://upload.wikimedia.org/wikipedia/commons/f/f0/Google_Bard_logo.svg",
    website: "https://bard.google.com",
    pricing: "Free",
    categories: ["Natural Language Processing", "Research"],
    tags: ["chatbot", "google", "information retrieval"],
    trending: true
  },
  {
    id: "17",
    name: "DeepL",
    description: "Neural machine translation service that provides more natural-sounding translations than many competitors.",
    logo: "https://static.deepl.com/img/logo/DeepL_Logo_darkBlue_v2.svg",
    website: "https://www.deepl.com",
    pricing: "Freemium",
    categories: ["Natural Language Processing", "Productivity"],
    tags: ["translation", "language", "writing assistant"]
  },
  {
    id: "18",
    name: "Adobe Firefly",
    description: "Creative generative AI tools for image generation and editing integrated with Adobe's creative suite.",
    logo: "https://upload.wikimedia.org/wikipedia/commons/e/e6/Adobe_Firefly_logo.svg",
    website: "https://www.adobe.com/products/firefly.html",
    pricing: "Paid",
    categories: ["Design", "Computer Vision"],
    tags: ["image generation", "creative tools", "adobe"],
    featured: true
  },
  {
    id: "19",
    name: "Otter.ai",
    description: "AI-powered meeting assistant that records, transcribes, and summarizes conversations in real-time.",
    logo: "https://upload.wikimedia.org/wikipedia/commons/2/20/Otter.ai_logo.svg",
    website: "https://otter.ai",
    pricing: "Freemium",
    categories: ["Productivity", "Audio/Video"],
    tags: ["transcription", "meeting notes", "audio"],
    trending: true
  },
  {
    id: "20",
    name: "Replicate",
    description: "Platform for running machine learning models in the cloud with a simple API.",
    logo: "https://replicate.com/static/favicon.e390e65f.png",
    website: "https://replicate.com",
    pricing: "Paid",
    categories: ["Developer Tools", "Research"],
    tags: ["model hosting", "api", "machine learning"]
  },
  {
    id: "21",
    name: "Whisper",
    description: "Open-source speech recognition system by OpenAI that approaches human-level robustness and accuracy.",
    logo: "https://upload.wikimedia.org/wikipedia/commons/a/a4/OpenAI_Logo.svg",
    website: "https://openai.com/research/whisper",
    pricing: "Free",
    categories: ["Audio/Video", "Natural Language Processing"],
    tags: ["speech recognition", "transcription", "open source"]
  },
  {
    id: "22",
    name: "Luma AI",
    description: "3D content creation platform that generates photorealistic 3D assets and scenes from text or images.",
    logo: "https://lumalabs.ai/favicon-32x32.png",
    website: "https://lumalabs.ai",
    pricing: "Freemium",
    categories: ["Design", "Computer Vision"],
    tags: ["3d generation", "photorealistic", "ai design"]
  },
  {
    id: "23",
    name: "Eleven Labs",
    description: "AI voice technology platform for generating natural-sounding voice content with emotional resonance.",
    logo: "https://storage.googleapis.com/eleven-public-prod/logos/eleven-labs-logo-60.png",
    website: "https://elevenlabs.io",
    pricing: "Freemium",
    categories: ["Audio/Video", "Natural Language Processing"],
    tags: ["voice synthesis", "text-to-speech", "ai voice"],
    featured: true
  },
  {
    id: "24",
    name: "AutoGPT",
    description: "Experimental open-source project that enables GPT models to autonomously achieve goals by breaking them into subtasks.",
    logo: "https://avatars.githubusercontent.com/u/128686189",
    website: "https://github.com/Significant-Gravitas/AutoGPT",
    pricing: "Free",
    categories: ["Developer Tools", "Research", "Productivity"],
    tags: ["autonomous ai", "agent", "open source"]
  }
];

export const getAllCategories = (): CategoryType[] => {
  if (!Array.isArray(tools)) return [];
  
  const categoriesSet = new Set<CategoryType>();
  
  tools.forEach(tool => {
    tool.categories.forEach(category => {
      categoriesSet.add(category);
    });
  });
  
  return Array.from(categoriesSet).sort();
};

export const getAllTags = (): string[] => {
  if (!Array.isArray(tools)) return [];
  
  const tagsSet = new Set<string>();
  
  tools.forEach(tool => {
    tool.tags.forEach(tag => {
      tagsSet.add(tag);
    });
  });
  
  return Array.from(tagsSet).sort();
};

export const getPricingOptions = (): PricingType[] => {
  if (!Array.isArray(tools)) return [];
  
  const pricingSet = new Set<PricingType>();
  
  tools.forEach(tool => {
    pricingSet.add(tool.pricing);
  });
  
  return Array.from(pricingSet).sort();
};

export const getToolsByCategory = (category: CategoryType): Tool[] => {
  if (!Array.isArray(tools)) return [];
  return tools.filter(tool => tool.categories.includes(category));
};

export const getFeaturedTools = (): Tool[] => {
  if (!Array.isArray(tools)) return [];
  return tools.filter(tool => tool.featured);
};

export const getTrendingTools = (): Tool[] => {
  if (!Array.isArray(tools)) return [];
  return tools.filter(tool => tool.trending);
};
