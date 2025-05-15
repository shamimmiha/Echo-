
import { SearchBar } from "@/components/SearchBar";
import { PricingFilter } from "@/components/PricingFilter";
import { PricingType } from "@/data/tools";

interface DirectoryHeaderProps {
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  selectedPricing: PricingType | null;
  setSelectedPricing: (pricing: PricingType | null) => void;
  toolsCount: number;
}

export function DirectoryHeader({ 
  searchQuery, 
  setSearchQuery,
  selectedPricing,
  setSelectedPricing,
  toolsCount
}: DirectoryHeaderProps) {
  return (
    <div className="mb-6">
      <div className="flex justify-between items-center mb-4">
        <h1 className="text-3xl font-bold">AI Tools Directory</h1>
        <div className="text-sm text-muted-foreground">
          Showing <span className="font-medium">{toolsCount}</span> tools
        </div>
      </div>
      
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="flex-grow">
          <SearchBar searchQuery={searchQuery} setSearchQuery={setSearchQuery} />
        </div>
        <div>
          <PricingFilter selectedPricing={selectedPricing} onSelectPricing={setSelectedPricing} />
        </div>
      </div>
    </div>
  );
}
