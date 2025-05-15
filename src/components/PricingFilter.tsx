
import { PricingType, getPricingOptions } from "@/data/tools";
import { Check, ChevronsUpDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { useState } from "react";

interface PricingFilterProps {
  selectedPricing: PricingType | null;
  onSelectPricing: (pricing: PricingType | null) => void;
}

export function PricingFilter({ selectedPricing, onSelectPricing }: PricingFilterProps) {
  const [open, setOpen] = useState(false);
  // Make sure we always have a valid array
  const pricingOptions = getPricingOptions() || [];

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className="justify-between"
          size="sm"
        >
          {selectedPricing || "Pricing"}
          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[200px] p-0">
        <Command>
          <CommandInput placeholder="Search pricing..." />
          <CommandEmpty>No pricing option found.</CommandEmpty>
          <CommandGroup>
            <CommandItem
              onSelect={() => {
                onSelectPricing(null);
                setOpen(false);
              }}
              className="cursor-pointer"
            >
              <Check
                className={cn(
                  "mr-2 h-4 w-4",
                  !selectedPricing ? "opacity-100" : "opacity-0"
                )}
              />
              All Pricing
            </CommandItem>
            {pricingOptions.map((pricing) => (
              <CommandItem
                key={pricing}
                onSelect={() => {
                  onSelectPricing(pricing);
                  setOpen(false);
                }}
                className="cursor-pointer"
              >
                <Check
                  className={cn(
                    "mr-2 h-4 w-4",
                    selectedPricing === pricing ? "opacity-100" : "opacity-0"
                  )}
                />
                {pricing}
              </CommandItem>
            ))}
          </CommandGroup>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
