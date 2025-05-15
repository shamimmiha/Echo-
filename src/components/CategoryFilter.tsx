
import { CategoryType, getAllCategories } from "@/data/tools";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface CategoryFilterProps {
  selectedCategory: CategoryType | "All";
  onSelectCategory: (category: CategoryType | "All") => void;
}

export function CategoryFilter({ selectedCategory, onSelectCategory }: CategoryFilterProps) {
  const categories = ["All", ...getAllCategories()];
  
  return (
    <div className="flex flex-wrap gap-2 mb-6">
      {categories.map((category) => (
        <Button 
          key={category} 
          variant={selectedCategory === category ? "default" : "outline"}
          size="sm"
          onClick={() => onSelectCategory(category as CategoryType | "All")}
          className={cn(
            selectedCategory === category ? "bg-primary text-primary-foreground" : "bg-transparent",
            "rounded-full"
          )}
        >
          {category}
        </Button>
      ))}
    </div>
  );
}
