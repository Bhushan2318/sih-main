import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ApiError } from "./api/client";
import { DashboardPage } from "./pages/DashboardPage";
import "./styles.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 15_000,
      refetchOnWindowFocus: false,

      // A free host sleeps after 15 idle minutes and takes 30-50s to wake. One retry a
      // second later cannot outlast that, so the first visitor to a cold site saw an
      // error for something that was about to work. These delays (1, 2, 4, 8, 10, 10s)
      // span the wake window instead.
      //
      // A 4xx is excluded because it is a real answer: the server understood and said no,
      // and asking six more times neither changes that nor is honest about it.
      retry: (failureCount, error) => {
        if (error instanceof ApiError && error.status >= 400 && error.status < 500) return false;
        return failureCount < 5;
      },
      retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 10_000),
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <DashboardPage />
    </QueryClientProvider>
  );
}
