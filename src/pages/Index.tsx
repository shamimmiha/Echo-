
import { useState, useEffect } from "react";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { CategoryType, PricingType, Tool, tools } from "@/data/tools";
import { DirectorySidebar } from "@/components/DirectorySidebar";
import { DirectoryHeader } from "@/components/DirectoryHeader";
import { CategoryFilter } from "@/components/CategoryFilter";
import { ToolsGrid } from "@/components/ToolsGrid";
import { FeaturedTools } from "@/components/FeaturedTools";
import { Button } from "@/components/ui/button";
import { ChevronUp } from "lucide-react";

const Index = () => {
  const [selectedCategory, setSelectedCategory] = useState<CategoryType | "All">("All");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedPricing, setSelectedPricing] = useState<PricingType | null>(null);
  const [filteredTools, setFilteredTools] = useState<Tool[]>(tools);
  const [showScrollToTop, setShowScrollToTop] = useState(false);

  // Apply filters when dependencies change
  useEffect(() => {
    let result = [...tools];

    // Filter by category
    if (selectedCategory !== "All") {
      result = result.filter((tool) => 
        tool.categories.includes(selectedCategory as CategoryType)
      );
    }

    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      result = result.filter(
        (tool) =>
          tool.name.toLowerCase().includes(query) ||
          tool.description.toLowerCase().includes(query) ||
          tool.tags.some((tag) => tag.toLowerCase().includes(query)) ||
          tool.categories.some((category) => category.toLowerCase().includes(query))
      );
    }

    // Filter by pricing
    if (selectedPricing) {
      result = result.filter((tool) => tool.pricing === selectedPricing);
    }

    setFilteredTools(result);
  }, [selectedCategory, searchQuery, selectedPricing]);

  // Scroll to top button logic
  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 300) {
        setShowScrollToTop(true);
      } else {
        setShowScrollToTop(false);
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  };

  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full">
        <DirectorySidebar 
          selectedCategory={selectedCategory} 
          onSelectCategory={setSelectedCategory} 
        />
        
        <div className="flex-1">
          <div className="container max-w-7xl py-6">
            <div className="flex justify-end mb-4">
              <SidebarTrigger />
            </div>
            
            <DirectoryHeader 
              searchQuery={searchQuery}
              setSearchQuery={setSearchQuery}
              selectedPricing={selectedPricing}
              setSelectedPricing={setSelectedPricing}
              toolsCount={filteredTools.length}
            />

            {/* Only display featured tools when no filters are applied */}
            {selectedCategory === "All" && !searchQuery && !selectedPricing && (
              <FeaturedTools />
            )}
            
            <CategoryFilter 
              selectedCategory={selectedCategory} 
              onSelectCategory={setSelectedCategory} 
            />
            
            {filteredTools.length > 0 ? (
              <ToolsGrid tools={filteredTools} />
            ) : (
              <div className="text-center py-16">
                <p className="text-xl font-medium text-muted-foreground">No tools found matching your filters</p>
                <Button
                  variant="link"
                  onClick={() => {
                    setSelectedCategory("All");
                    setSearchQuery("");
                    setSelectedPricing(null);
                  }}
                  className="mt-2"
                >
                  Reset all filters
                </Button>
              </div>
            )}
          </div>
        </div>

        {/* Scroll to top button */}
        {showScrollToTop && (
          <Button
            size="icon"
            className="fixed bottom-6 right-6 rounded-full shadow-lg"
            onClick={scrollToTop}
          >
            <ChevronUp className="h-5 w-5" />
          </Button>
        )}
      </div>
    </SidebarProvider>
  );
};

export default Index;
