
import { CategoryType, getAllCategories } from "@/data/tools";
import { cn } from "@/lib/utils";
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
} from "@/components/ui/sidebar";

interface DirectorySidebarProps {
  selectedCategory: CategoryType | "All";
  onSelectCategory: (category: CategoryType | "All") => void;
}

export function DirectorySidebar({ selectedCategory, onSelectCategory }: DirectorySidebarProps) {
  const categories = getAllCategories();

  return (
    <Sidebar>
      <SidebarHeader>
        <div className="font-bold text-lg p-4">AI Tools Directory</div>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Categories</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton
                  onClick={() => onSelectCategory("All")}
                  className={cn(
                    selectedCategory === "All" &&
                      "bg-sidebar-accent text-sidebar-accent-foreground"
                  )}
                >
                  <span>All Tools</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
              
              {categories.map((category) => (
                <SidebarMenuItem key={category}>
                  <SidebarMenuButton
                    onClick={() => onSelectCategory(category)}
                    className={cn(
                      selectedCategory === category &&
                        "bg-sidebar-accent text-sidebar-accent-foreground"
                    )}
                  >
                    <span>{category}</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
        
        <SidebarGroup>
          <SidebarGroupLabel>About</SidebarGroupLabel>
          <SidebarGroupContent className="p-4 text-sm text-muted-foreground">
            <p>This is a comprehensive directory of AI tools that lets you explore various tools across different categories.</p>
            <p className="mt-2">Click on a category to filter the tools or use the search bar to find specific tools.</p>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}
