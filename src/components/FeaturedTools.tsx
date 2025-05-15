
import { getFeaturedTools, Tool } from "@/data/tools";
import { ToolCard } from "@/components/ToolCard";
import { Separator } from "@/components/ui/separator";

interface FeaturedToolsProps {
  tools?: Tool[];
}

export function FeaturedTools({ tools = getFeaturedTools() }: FeaturedToolsProps) {
  return (
    <div className="mb-10 animate-fade-in">
      <h2 className="text-2xl font-bold mb-4">Featured Tools</h2>
      <Separator className="mb-6" />
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {tools.map((tool) => (
          <ToolCard key={tool.id} tool={tool} />
        ))}
      </div>
    </div>
  );
}
