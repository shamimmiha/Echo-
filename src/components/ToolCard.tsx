
import { Tool } from "@/data/tools";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ExternalLink } from "lucide-react";

interface ToolCardProps {
  tool: Tool;
}

export function ToolCard({ tool }: ToolCardProps) {
  return (
    <Card className="tool-card overflow-hidden h-full flex flex-col">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded overflow-hidden flex items-center justify-center bg-white">
              <img 
                src={tool.logo} 
                alt={`${tool.name} logo`} 
                className="w-full h-full object-contain"
                onError={(e) => {
                  (e.target as HTMLImageElement).src = "/placeholder.svg";
                }}
              />
            </div>
            <CardTitle className="text-lg">{tool.name}</CardTitle>
          </div>
          {(tool.featured || tool.trending) && (
            <div>
              {tool.featured && <Badge variant="default" className="bg-purple-500 hover:bg-purple-600 ml-1">Featured</Badge>}
              {tool.trending && <Badge variant="secondary" className="bg-blue-500 text-white hover:bg-blue-600 ml-1">Trending</Badge>}
            </div>
          )}
        </div>
        <div className="mt-2">
          <Badge variant="outline" className="mr-1">{tool.pricing}</Badge>
          {tool.categories.slice(0, 2).map((category) => (
            <Badge key={category} variant="secondary" className="mr-1">{category}</Badge>
          ))}
        </div>
      </CardHeader>
      <CardContent className="pb-2 flex-grow">
        <CardDescription className="line-clamp-3">{tool.description}</CardDescription>
        <div className="mt-3 flex flex-wrap gap-1">
          {tool.tags.slice(0, 3).map((tag) => (
            <span key={tag} className="text-xs bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded-sm">
              {tag}
            </span>
          ))}
        </div>
      </CardContent>
      <CardFooter>
        <Button asChild className="w-full" size="sm">
          <a href={tool.website} target="_blank" rel="noopener noreferrer" className="flex items-center justify-center">
            <span>Visit Site</span>
            <ExternalLink className="ml-2 h-4 w-4" />
          </a>
        </Button>
      </CardFooter>
    </Card>
  );
}
