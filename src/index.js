import { env as workerEnv } from "cloudflare:workers";
import {
    Container,
    getContainer
} from "@cloudflare/containers";


export class VeritasContainer extends Container {

    defaultPort = 8080;

    // Keep the BERT models loaded for a while
    // after the last request.
    sleepAfter = "10m";

    // VERITAS needs internet access for:
    // Neon, Google Fact Check, Gmail SMTP
    // and article extraction.
    enableInternet = true;

    // Flask health check
    pingEndpoint = "localhost/ping";


    // Pass Cloudflare secrets into the
    // Python container as environment variables.
    envVars = {

        DATABASE_URL:
            workerEnv.DATABASE_URL,

        SECRET_KEY:
            workerEnv.SECRET_KEY,

        GOOGLE_FACTCHECK_API_KEY:
            workerEnv.GOOGLE_FACTCHECK_API_KEY,

        SMTP_USERNAME:
            workerEnv.SMTP_USERNAME,

        SMTP_PASSWORD:
            workerEnv.SMTP_PASSWORD,

        SMTP_FROM:
            workerEnv.SMTP_FROM,

        SMTP_HOST:
            "smtp.gmail.com",

        SMTP_PORT:
            "587",

        COOKIE_SECURE:
            "1"
    };


    onStart() {

        console.log(
            "VERITAS container started"
        );

    }


    onStop() {

        console.log(
            "VERITAS container stopped"
        );

    }


    onError(error) {

        console.error(
            "VERITAS container error:",
            error
        );

        throw error;

    }
}


export default {

    async fetch(request, env) {

        const container = getContainer(
            env.VERITAS_CONTAINER,
            "veritas-main"
        );

        return container.fetch(
            request
        );

    }

};