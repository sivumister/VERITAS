import { Container } from "@cloudflare/containers";

export class VeritasContainer extends Container {
    defaultPort = 8080;

    // Keep VERITAS warm for 10 minutes after its last request.
    // This avoids reloading the BERT models for every request.
    sleepAfter = "10m";

    onStart() {
        console.log("VERITAS container started");
    }

    onStop() {
        console.log("VERITAS container stopped");
    }

    onError(error) {
        console.error("VERITAS container error:", error);
        throw error;
    }
}

export default {
    async fetch(request, env) {
        const container =
            env.VERITAS_CONTAINER.getByName("veritas-main");

        return container.fetch(request);
    }
};